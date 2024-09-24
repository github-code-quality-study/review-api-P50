import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

def filter_by_date_range(filtered_reviews, query_params):
    """Filters reviews based on provided date range.

    Args:
        filtered_reviews: List of reviews to filter.
        query_params: Query parameters containing 'start_date' and/or 'end_date'.

    Returns:
        Filtered list of reviews.
    """

    if "start_date" in query_params:
        try:
            start_date = datetime.datetime.strptime(query_params["start_date"][0], "%Y-%m-%d")
        except ValueError:
            # Handle invalid start_date format
            return filtered_reviews  # Or return an error message

    if "end_date" in query_params:
        try:
            end_date = datetime.datetime.strptime(query_params["end_date"][0], "%Y-%m-%d")
        except ValueError:
            # Handle invalid end_date format
            return filtered_reviews  # Or return an error message

    if "start_date" in query_params and "end_date" in query_params:
        filtered_reviews = [
            review
            for review in filtered_reviews
            if start_date <= datetime.datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") <= end_date
        ]
    elif "start_date" in query_params:
        filtered_reviews = [
            review
            for review in filtered_reviews
            if start_date <= datetime.datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S")
        ]
    elif "end_date" in query_params:
        filtered_reviews = [
            review
            for review in filtered_reviews
            if datetime.datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") <= end_date
        ]

    return filtered_reviews

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information   
        such as: REQUEST_METHOD, CONTENT_LENGTH,   
        QUERY_STRING,
                PATH_INFO, CONTENT_TYPE, etc.
                """

        if environ["REQUEST_METHOD"] == "GET":
            # Get query parameters
            query_params = parse_qs(environ["QUERY_STRING"])
            filtered_reviews = reviews.copy()  # Copy to avoid modifying original data

            # Filter by location (if provided)
            if "location" in query_params:
                location = query_params["location"][0]
                filtered_reviews = [
                    review for review in filtered_reviews if review["Location"] == location
                ]

            filtered_reviews = filter_by_date_range(filtered_reviews, query_params)
                
            # Allowed locations
            allowed_locations = """Albuquerque, New Mexico
Carlsbad, California
Chula Vista, California
Colorado Springs, Colorado
Denver, Colorado
El Cajon, California
El Paso, Texas
Escondido, California
Fresno, California
La Mesa, California
Las Vegas, Nevada
Los Angeles, California
Oceanside, California
Phoenix, Arizona
Sacramento, California
Salt Lake City, Utah
Salt Lake City, Utah
San Diego, California
Tucson, Arizona""".splitlines()
            
            if "location" in query_params and query_params["location"][0] not in allowed_locations:
                filtered_reviews = []  # Filter out reviews from non-allowed locations

            # Analyze sentiment and sort by compound score
            for review in filtered_reviews:
                review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])
            filtered_reviews.sort(key=lambda x: x["sentiment"]["compound"], reverse=True)
            # Convert to JSON response
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body))),
            ])

            return [response_body]

        elif environ["REQUEST_METHOD"] == "POST":
            # Get request body data
            request_body = environ["wsgi.input"].read().decode("utf-8")
            parsed_body = parse_qs(request_body)
            try:
                # Extract data from request body
                review_body = parsed_body["ReviewBody"][0]
                location = parsed_body["Location"][0]
                
            except:
                response_body = json.dumps({"error": "ReviewBody and Location are required"}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body))),
                ])
                return [response_body]

            # Validate location
            allowed_locations = """Albuquerque, New Mexico
Carlsbad, California
Chula Vista, California
Colorado Springs, Colorado
Denver, Colorado
El Cajon, California
El Paso, Texas
Escondido, California
Fresno, California
La Mesa, California
Las Vegas, Nevada
Los Angeles, California
Oceanside, California
Phoenix, Arizona
Sacramento, California
Salt Lake City, Utah
Salt Lake City, Utah
San Diego, California
Tucson, Arizona""".splitlines()


            if location not in allowed_locations:
                response_body = json.dumps({"error": "Invalid location provided"}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body))),
                ])
                return [response_body]

            # Create new review data
            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": self.analyze_sentiment(review_body),
            }
            # Add new review to list
            reviews.append(new_review)

            # Convert to JSON response
            response_body = json.dumps(new_review, indent=2).encode("utf-8")

            # Set response headers
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body))),
            ])

            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()