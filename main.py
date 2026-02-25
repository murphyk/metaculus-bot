import argparse
import asyncio
import logging
from datetime import datetime, timezone
import dotenv
from typing import Literal

# nest_asyncio is incompatible with Python 3.14 (breaks asyncio.current_task()).
# On 3.14+ we disable nest_asyncio and instead patch asyncio.run so that calls
# from inside a running event loop (as forecasting_tools does internally) are
# dispatched to a fresh thread rather than failing with RuntimeError.
# On earlier versions, forecasting_tools needs nest_asyncio for the same reason.
import sys
import threading
import nest_asyncio
if sys.version_info >= (3, 14):
    nest_asyncio.apply = lambda *a, **kw: None
    _orig_asyncio_run = asyncio.run

    def _asyncio_run_nested_safe(coro, /, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _orig_asyncio_run(coro, **kwargs)
        # Already inside a running loop — run in a new thread with its own loop
        _result: list = [None]
        _exc: list = [None]
        def _target():
            try:
                _result[0] = _orig_asyncio_run(coro)
            except BaseException as e:
                _exc[0] = e
        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join()
        if _exc[0] is not None:
            raise _exc[0]
        return _result[0]

    asyncio.run = _asyncio_run_nested_safe


from forecasting_tools import (
    ApiFilter,
    AskNewsSearcher,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    SmartSearcher,
    clean_indents,
)

from backtest_pipeline import generate_html, migrate_reports_json, parse_forecasts_to_preds, save_bot_config, save_raw_forecasts, save_truth
from predict_binary import BinaryForecastMixin
from predict_multiple_choice import MultipleChoiceForecastMixin
from predict_numeric import NumericForecastMixin
from predict_date import DateForecastMixin
from predict_conditional import ConditionalForecastMixin
from predict_shared import SharedForecastHelpers

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class SpringTemplateBot2026(
    BinaryForecastMixin,
    MultipleChoiceForecastMixin,
    NumericForecastMixin,
    DateForecastMixin,
    ConditionalForecastMixin,
    SharedForecastHelpers,
    ForecastBot,
):
    """
    This is the template bot for Spring 2026 Metaculus AI Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    Main changes since Fall:
    - Additional prompting has been added to numeric questions to emphasize putting pecentile values in the correct order.
    - Support for conditional and date questions has been added
    - Note: Spring AIB will not use date/conditional questions, so these are only for forecasting on the main site as you wish.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament_current", "tournament_fall_2025", "metaculus_cup", "test_questions"],
        default="tournament_current",
        help="Specify the run mode (default: tournament_current)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament_current", "tournament_fall_2025", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament_current",
        "tournament_fall_2025",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = SpringTemplateBot2026(  # type: ignore[abstract]
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(model="openrouter/anthropic/claude-sonnet-4-6", temperature=0.3),
            "summarizer": GeneralLlm(model="openrouter/anthropic/claude-haiku-4-5", temperature=0.3),
            "researcher": "smart-searcher/openrouter/anthropic/claude-sonnet-4-6",
            #"researcher": GeneralLlm(model="openrouter/openai/gpt-4o-search-preview", temperature=0.1),
            "parser": GeneralLlm(model="openrouter/anthropic/claude-haiku-4-5", temperature=0.3),
        },
    )

    BOT_NAME = "spring_template_2026"
    client = MetaculusClient()
    forecast_reports: list = []
    if run_mode == "tournament_current":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "tournament_fall_2025":
        # Backtest mode: questions are already resolved.
        # forecast_on_tournament only fetches open questions, so fetch resolved ones directly.
        template_bot.publish_reports_to_metaculus = False
        template_bot.folder_to_save_reports_to = "fall_2025_reports"
        save_bot_config(template_bot, BOT_NAME)
        api_filter = ApiFilter(
            allowed_statuses=["resolved"],
            allowed_tournaments=[client.AIB_FALL_2025_ID],
        )
        fall_2025_questions = asyncio.run(
            client.get_questions_matching_filter(api_filter)
        )
        save_truth(fall_2025_questions, "fall_2025")
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(fall_2025_questions, return_exceptions=True)
        )
        save_raw_forecasts(forecast_reports, BOT_NAME, "fall_2025")
        parse_forecasts_to_preds(BOT_NAME, "fall_2025")
        generate_html("fall_2025", output_html="fall_2025_backtest.html")
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            client.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
