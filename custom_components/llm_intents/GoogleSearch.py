import logging
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util.json import JsonObjectType

from .cache import SQLiteCache
from .const import (
    CONF_GOOGLE_SEARCH_API_KEY,
    CONF_GOOGLE_SEARCH_MODEL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


class GoogleSearchTool(llm.Tool):
    """Tool for searching the web via Google Search Grounding (Gemini)."""

    name = "search_web_google"
    description = (
        "Search the web for up-to-date information about the world, "
        "news, or facts not present in your knowledge base."
    )

    # Instruction for the model on how to handle results
    response_instruction = """
    Используй результаты поиска и текущую дату из системного промпта. 
    Если пользователь спросил "через сколько дней/месяцев", ты ОБЯЗАНА сама вычислить разницу, 
    основываясь на сегодняшнем числе.

    ТВОЯ ЗАДАЧА:
    Отвечать живым, естественным языком (лирика и комментарии разрешены), НО с хирургической точностью к фактам.

    СТРОГИЕ ПРАВИЛА ОБРАБОТКИ ФАКТОВ:
    1. ИЗОЛЯЦИЯ ЛОКАЦИЙ (Самое важное): Если в тексте упомянуто несколько мест (например, Индия, Казахстан, Камчатка), ты обязана перечислить их ОТДЕЛЬНО.
       - ЗАПРЕЩЕНО смешивать данные: Нельзя писать "В Казахстане и на Камчатке были толчки до 5 баллов", если 5 баллов было только на Камчатке.
       - Правильно: "В Казахстане — 4.2. А вот на Камчатке — до 5.0".
    
    2. ЗАПРЕТ НА "КРАЖУ" ЦИФР: Никогда не приписывай магнитуду или дату одного события другому. Проверяй каждое число: к какому городу оно относится в исходнике?
    
    3. ГЕОГРАФИЯ: Не меняй названия мест. Если написано "Южная Атлантика", не пиши "Америка".

    Итог: Говори красиво, но цифры и города раскладывай строго по полочкам, не смешивая их.
    """

    parameters = vol.Schema(
        {
            vol.Required("query", description="The search query"): str,
        }
    )

    def wrap_response(self, response: dict) -> dict:
        response["instruction"] = self.response_instruction
        return response

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Call the search tool."""
        config_data = hass.data[DOMAIN].get("config", {})
        entry = next(iter(hass.config_entries.async_entries(DOMAIN)))
        config_data = {**config_data, **entry.options}

        query = tool_input.tool_args["query"]

        api_key = config_data.get(CONF_GOOGLE_SEARCH_API_KEY)
        model = config_data.get(CONF_GOOGLE_SEARCH_MODEL, "gemini-2.0-flash-exp")

        if not api_key:
            _LOGGER.error("Google Search API key not configured")
            return {"error": "Google Search API key not configured"}

        _LOGGER.info("Google Search requested for: %s", query)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [{"parts": [{"text": query}]}],
            "tools": [{"google_search": {}}],
            "systemInstruction": {
                "parts": [
                    {
                        "text": "Ты — поисковый ассистент. Используй поиск для получения актуальной информации. Верни краткую выжимку по результатам поиска. Всегда отвечай на русском языке."
                    }
                ]
            },
        }

        try:
            cache = SQLiteCache()
            # We use the model and query as key for cache to distinguish between models if needed
            cache_params = {"model": model, "query": query}
            cached_response = cache.get(__name__, cache_params)

            if cached_response:
                return self.wrap_response(cached_response)

            session = async_get_clientsession(hass)
            async with session.post(url, json=payload, timeout=15) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    _LOGGER.error("Gemini API error: %s - %s", resp.status, text)
                    return {"error": f"Search error: status {resp.status}"}

                data = await resp.json()

                # Extract text and sources (Grounding Metadata)
                candidates = data.get("candidates", [])
                if not candidates:
                    return {"results": "Информация не найдена."}

                candidate = candidates[0]
                content = (
                    candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
                )

                # Extract source links
                metadata = candidate.get("groundingMetadata", {})
                attributions = metadata.get("groundingAttributions", [])

                sources = []
                for attr in attributions:
                    web_source = attr.get("web", {})
                    if web_source:
                        sources.append(
                            {
                                "title": web_source.get("title"),
                                "url": web_source.get("uri"),
                            }
                        )

                if not content and not sources:
                    return {"results": "No information found."}

                response = {
                    "answer_summary": content,
                    "sources": sources[:3],  # Limit to 3 sources for conciseness
                }

                cache.set(__name__, cache_params, response)
                return self.wrap_response(response)

        except Exception as e:
            _LOGGER.error("Error performing Google Search: %s", e)
            return {"error": f"An error occurred during search: {e!s}"}
