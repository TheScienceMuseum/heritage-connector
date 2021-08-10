import json

def etc_uri(system_number):
  return f"http://collections.vam.ac.uk/item/{system_number}"

# This needs to go through to the contextual page or authority api endpoint when avaialable
def etc_search_uri(query_param, system_number):
  return f"https://api.vam.ac.uk/v2/objects/search?{query_param}={system_number}"

def join_record(obj_system_number, join_index, join_system_number, join_type):
  row = {'URI_1': etc_uri(obj_system_number), 'URI_2': etc_search_uri(join_index, join_system_number), 'relationship': join_type}
  return row

def events_transforming(doc):
  """Creates a flattened object with minimal fields for conversion to data frame

  Args:
      record ([object]): a single record from our etc eleasticsearch events index
  """
  hc_record = {}
  hc_record['URI'] = etc_search_uri('id_event', doc['systemNumber'])
  hc_record['SYSTEM_NUMBER'] = doc['systemNumber']

  if doc['name']:
    hc_record['NAME'] = doc['name']
  else:
    hc_record['NAME'] = ""

  if doc['date']['earliest']:
    hc_record['DATE_EARLIEST'] = doc['date']['earliest']
  else:
    hc_record['DATE_EARLIEST'] = None

  if doc['date']['latest']:
    hc_record['DATE_LATEST'] = doc['date']['latest']
  else:
    hc_record['DATE_LATEST'] = None

  return hc_record


def persons_transforming(doc):
  """Creates a flattened object with minimal fields for conversion to data frame

  Args:
      record ([object]): a single record from our etc eleasticsearch persons index
  """
  hc_record = {}
  hc_record['URI'] = etc_search_uri('id_person', doc['systemNumber'])
  hc_record['SYSTEM_NUMBER'] = doc['systemNumber']

  if doc['title']:
    hc_record['TITLE_NAME'] = doc['title']
  else:
    hc_record['TITLE_NAME'] = ""

  if doc['foreNames']:
    hc_record['FORENAME'] = doc['foreNames']
  else:
    hc_record['FORENAME'] = ""

  if doc['surNames']:
    hc_record['SURNAME'] = doc['surNames']
  else:
    hc_record['SURNAME'] = ""

  if doc['naturalName']:
    hc_record['NATURAL_NAME'] = doc['naturalName']
  else:
    hc_record['NATURAL_NAME'] = ""

  if doc['birthDate']['earliest']:
    hc_record['BIRTHDATE_EARLIEST'] = doc['birthDate']['earliest']
  else:
    hc_record['BIRTHDATE_EARLIEST'] = None

  if doc['birthDate']['latest']:
    hc_record['BIRTHDATE_LATEST'] = doc['birthDate']['latest']
  else:
    hc_record['BIRTHDATE_LATEST'] = None

  if doc['birthPlace']['text']:
    hc_record['BIRTHPLACE'] = doc['birthPlace']['text']
  else:
    hc_record['BIRTHPLACE'] = ""

  if doc['deathPlace']['text']:
    hc_record['DEATHPLACE'] = doc['deathPlace']['text']
  else:
    hc_record['DEATHPLACE'] = ""

  if doc['nationality']:
    hc_record['NATIONALITY'] = doc['nationality']
  else:
    hc_record['NATIONALITY'] = ""

  if doc['biography']:
    hc_record['BIOGRAPHY'] = doc['biography']
  else:
    hc_record['BIOGRAPHY'] = ""

  return hc_record

def objects_transforming(doc):
  """Creates a flattened object with minimal fields for conversion to data frame

  Args:
      record ([object]): a single record from our etc eleasticsearch objects index
  """
  hc_record = {}
  hc_record['URI'] = etc_uri(doc['systemNumber'])
  hc_record['SYSTEM_NUMBER'] = doc['systemNumber']

  if doc['_primaryTitle']:
    hc_record['PRIMARY_TITLE'] = doc['_primaryTitle']
  else:
    hc_record['PRIMARY_TITLE'] = ""

  if doc['_primaryPlace']:
    hc_record['PRIMARY_PLACE'] = doc['_primaryPlace']
  else:
    hc_record['PRIMARY_PLACE'] = None

  if doc['_primaryDate']:
    hc_record['PRIMARY_DATE'] = doc['_primaryDate']
  else:
    hc_record['PRIMARY_DATE'] = None

  if doc['objectType']:
    hc_record['OBJECT_TYPE'] = doc['objectType']
  else:
    hc_record['OBJECT_TYPE'] = ""
  
  if doc['summaryDescription']:
    hc_record['DESCRIPTION'] = doc['summaryDescription']
  else:
    hc_record['DESCRIPTION'] = ""

  if doc['physicalDescription']:
    hc_record['PHYS_DESCRIPTION'] = doc['physicalDescription']
  else:
    hc_record['PHYS_DESCRIPTION'] = ""

  hc_record['ACCESSION_NUMBER'] = doc['accessionNumber']

  hc_record['COLLECTION'] = doc['_flatCollectionCodeTextId']

  if doc['_flatProductionTypesTextId']:
    hc_record['PRODUCTION_TYPE'] = doc['_flatProductionTypesTextId']
  else:
    hc_record['PRODUCTION_TYPE'] = ""
    

  return hc_record

def organisations_transforming(doc):
  """Creates a flattened object with minimal fields for conversion to data frame

  Args:
      record ([object]): a single record from our etc eleasticsearch organisations index
  """
  hc_record = {}
  hc_record['URI'] = etc_search_uri('id_organisation', doc['systemNumber'])
  hc_record['SYSTEM_NUMBER'] = doc['systemNumber']

  if doc['displayName']:
    hc_record['DISPLAY_NAME'] = doc['displayName']
  else:
    hc_record['DISPLAY_NAME'] = ""

  if doc['foundationDate']['earliest']:
    hc_record['FOUNDATION_DATE_EARLIEST'] = doc['foundationDate']['earliest']
  else:
    hc_record['FOUNDATION_DATE_EARLIEST'] = None

  if doc['foundationDate']['latest']:
    hc_record['FOUNDATION_DATE_LATEST'] = doc['foundationDate']['latest']
  else:
    hc_record['FOUNDATION_DATE_LATEST'] = None

  if doc['foundationPlace']['place']['text']:
    hc_record['FOUNDATION_PLACE_NAME'] = doc['foundationPlace']['place']['text']
  else:
    hc_record['FOUNDATION_PLACE_NAME'] = ""

  if doc['foundationPlace']['place']['id']:
    hc_record['FOUNDATION_PLACE_ID'] = doc['foundationPlace']['place']['id']
  else:
    hc_record['FOUNDATION_PLACE_ID'] = ""

  if doc['history']:
    hc_record['HISTORY'] = doc['history']
  else:
    hc_record['HISTORY'] = ""

  return hc_record


def object_joins(doc):
  """Generates potential joins with persons and organisations

  Args:
      record ([object]): a single record from our etc eleasticsearch persons index
  """

  # This needs to go through to the contextual page or authority api endpoint when avaialable at the moment it is linking to page of related objects
  join_list = []
  persons = [join_record(doc['systemNumber'], 'id_person', person['name']['id'], 'made_by') for person in doc['artistMakerPerson'] if person['name']['id']]
  content_persons = [join_record(doc['systemNumber'], 'id_person', person['id'], 'depicts') for person in doc['contentPerson'] if person['id']]
  assoc_persons = [join_record(doc['systemNumber'], 'id_person', person['id'], 'associated_with') for person in doc['associatedPerson'] if person['id']]
  orgs = [join_record(doc['systemNumber'], 'id_organisation', org['name']['id'], 'manufactured_by') for org in doc['artistMakerOrganisations'] if org['name']['id']]
  content_org = [join_record(doc['systemNumber'], 'id_organisation', org['id'], 'depicts') for org in doc['contentOrganisations'] if org['id']]
  assoc_org = [join_record(doc['systemNumber'], 'id_organisation', org['id'], 'associated_with') for org in doc['associatedOrganisations'] if org['id']]
  material = [join_record(doc['systemNumber'], 'id_material', mat['id'], 'made_from_material') for mat in doc['materials'] if mat['id']]
  technique = [join_record(doc['systemNumber'], 'id_technique', tech['id'], 'fabrication_method') for tech in doc['techniques'] if tech['id']]
  content_event = [join_record(doc['systemNumber'], 'id_event', event['id'], 'significant_event') for event in doc['associatedEvents'] if event['id']]
  assoc_event = [join_record(doc['systemNumber'], 'id_event', event['id'], 'significant_event') for event in doc['contentEvents'] if event['id']]

  join_list.extend(persons)
  join_list.extend(content_persons)
  join_list.extend(assoc_persons)
  join_list.extend(orgs)
  join_list.extend(content_org)
  join_list.extend(assoc_org)
  join_list.extend(material)
  join_list.extend(technique)
  join_list.extend(content_event)
  join_list.extend(assoc_event)
  return join_list
