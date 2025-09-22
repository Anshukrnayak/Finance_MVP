import logging
import json
from django.http import JsonResponse
from ratelimit.exceptions import Ratelimited

logger = logging.getLogger(__name__)

class GlobalExceptionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_exception(self, request, exception):
        if isinstance(exception, Ratelimited):
            return JsonResponse(
                {"error": "Rate limit exceeded"},
                status=429
            )

        logger.error(f"Unhandled exception: {exception}", exc_info=True)

        return JsonResponse(
            {"error": "Internal server error"},
            status=500
        )

    