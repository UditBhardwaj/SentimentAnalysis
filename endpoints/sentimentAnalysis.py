from flask import request, jsonify
from flask_restful import Resource
from .utils.model import SentimentAnalysisModel

class sentimentAnalysis(Resource):
  def __init__(self):
    self.model = SentimentAnalysisModel()
  def post(self):
    body = request.json
    reviews = body['reviews']
    predictions = self.model.classify(reviews)
    return {'predictions':predictions}