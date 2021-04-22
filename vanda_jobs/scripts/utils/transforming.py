import json

def etc_uri(system_number):
  return f"http://collections.vam.ac.uk/item/{system_number}"

def beta_api_uri(index, system_number):
  return f"http://api.vam.ac.uk/v2/{index}/{system_number}"

def join_record(obj_system_number, join_index, join_system_number, join_type):
  row = {'URI_1': etc_uri(obj_system_number), 'URI_2': beta_api_uri(join_index, join_system_number), 'relationship': join_type}
  return row

def persons_transforming(doc):
  """Creates a flattened object with minimal fields for conversion to data frame

  Args:
      record ([object]): a single record from our etc eleasticsearch persons index
  """
  hc_record = {}
  hc_record['URI'] = beta_api_uri('person', doc['systemNumber'])
  hc_record['SYSTEM_NUMBER'] = doc['systemNumber']

  if doc['title']:
    hc_record['TITLE_NAME'] = doc['title']
  else:
    hc_record['TITLE_NAME'] = None

  if doc['foreNames']:
    hc_record['FORENAME'] = doc['foreNames']
  else:
    hc_record['FORENAME'] = None

  if doc['surNames']:
    hc_record['SURNAME'] = doc['surNames']
  else:
    hc_record['SURNAME'] = None

  if doc['naturalName']:
    hc_record['NATURAL_NAME'] = doc['naturalName']
  else:
    hc_record['NATURAL_NAME'] = None

  if doc['birthDate']['earliest']:
    hc_record['BIRTHDATE_EARLIEST'] = doc['birthDate']['earliest']
  else:
    hc_record['BIRTHDATE_EARLIEST'] = None

  if doc['birthDate']['latest']:
    hc_record['BIRTHDATE_LATEST'] = doc['birthDate']['latest']
  else:
    hc_record['BIRTHDATE_LATEST'] = None

  if doc['nationality']:
    hc_record['NATIONALITY'] = doc['nationality']
  else:
    hc_record['NATIONALITY'] = None

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
    hc_record['PRIMARY_TITLE'] = None

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
    hc_record['OBJECT_TYPE'] = None
  
  if doc['summaryDescription']:
    hc_record['DESCRIPTION'] = doc['summaryDescription']
  else:
    hc_record['DESCRIPTION'] = None
    

  return hc_record

def organisations_transforming(doc):
  """Creates a flattened object with minimal fields for conversion to data frame

  Args:
      record ([object]): a single record from our etc eleasticsearch organisations index
  """
  hc_record = {}
  hc_record['URI'] = beta_api_uri('organisation', doc['systemNumber'])
  hc_record['SYSTEM_NUMBER'] = doc['systemNumber']

  if doc['displayName']:
    hc_record['DISPLAY_NAME'] = doc['displayName']
  else:
    hc_record['DISPLAY_NAME'] = None

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
    hc_record['FOUNDATION_PLACE_NAME'] = None

  if doc['foundationPlace']['place']['id']:
    hc_record['FOUNDATION_PLACE_ID'] = doc['foundationPlace']['place']['id']
  else:
    hc_record['FOUNDATION_PLACE_ID'] = None

  return hc_record


def object_joins(doc):
  """Generates potential joins with persons and organisations

  Args:
      record ([object]): a single record from our etc eleasticsearch persons index
  """
  join_list = []
  persons = [join_record(doc['systemNumber'], 'persons', person['name']['id'], 'made_by') for person in doc['artistMakerPerson'] if person['name']['id']]
  content_persons = [join_record(doc['systemNumber'], 'persons', person['id'], 'depicts') for person in doc['contentPerson'] if person['id']]
  assoc_persons = [join_record(doc['systemNumber'], 'persons', person['id'], 'associated_with') for person in doc['associatedPerson'] if person['id']]
  orgs = [join_record(doc['systemNumber'], 'organisations', org['name']['id'], 'manufactured_by') for org in doc['artistMakerOrganisations'] if org['name']['id']]
  content_org = [join_record(doc['systemNumber'], 'organisations', org['id'], 'depicts') for org in doc['contentOrganisations'] if org['id']]
  assoc_org = [join_record(doc['systemNumber'], 'organisations', org['id'], 'associated_with') for org in doc['associatedOrganisations'] if org['id']]
  
  join_list.extend(persons)
  join_list.extend(content_persons)
  join_list.extend(assoc_persons)
  join_list.extend(orgs)
  join_list.extend(content_org)
  join_list.extend(assoc_org)
  return join_list
