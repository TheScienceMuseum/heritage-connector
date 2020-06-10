import logging
import flask
import spacy
import requests

from spacy import displacy
from flask import request, jsonify, render_template
from markupsafe import escape
from lxml import html

from pprint import pprint


application = flask.Flask(__name__)
nlp = spacy.load("en_core_web_sm")


def fetchandscan(id):
    page = requests.get(f'https://collection.sciencemuseumgroup.org.uk/objects/{id}')
    tree = html.fromstring(page.content)
    description = tree.xpath('//div[@class="columns medium-8"]/p//text()')[0].strip()
    application.logger.info(description)
    doc = nlp(description)
    return displacy.render(doc, style="ent", minify=True)


@application.route('/')
def home():
    return render_template("home.html")


@application.route('/scan')
def scan():
    id = request.args.get('id')
    return fetchandscan(id)


if __name__ == '__main__':
    application.run(debug=True)
