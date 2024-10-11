# Bibliotecas

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import RSLPStemmer

# Baixar os recursos necessários para o uso do nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# instanciamos o Stemmer

stemmer = RSLPStemmer()

"""**Intenção**: objetivo principal da sentença, ou seja, ação do usuário

**Entidade**: são informações específicas que complementam a intenção.

Sentença = "Vamos pedir uma pizza hoje?"

Separação:
- intenção: pedir
- entidade: pizza, hoje

Ou:
- intenção: pedir pizza
- entidade: hoje
"""

def preprocessing(text):
  stop_words = set(stopwords.words('portuguese'))
  tokens = word_tokenize(text)
  filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
  stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
  return stemmed_tokens

intentions = {
    'pedir': ['quer', 'env', 'ped', 'peç', 'me v'],
    'consultar': ['ond', 'cheg', 'caminh', 'envi', 'disponibil'],
    'preço': ['prec', 'preç']
}

entities = {
    'refrigerante': ['refriger', 'peps', 'guar', 'fant', 'dolly', 'vedet', 'tubain', 'sod', 'sprit'],
    'cerveja': ['cervej', 'bav', 'skol', 'brahm', 'antar', 'heineken', 'budweis', 'stell', 'cryst'],
    'suco': ['suc', 'ade', 'prat', 'del vall', 'maguary', 'natur one', 'aur', 'camp larg', 'tial'],
    'gelo': ['gel', 'aro', 'qualita', 'litor', 'pinguim', 'rei', 'paragel'],
    'agua': ['agu', 'cryst', 'nestl', 'evian', 'sferri', 'vos', 'bonafont', 'minalb', 'lindoy']
}

responses = {
    'pedir': {
        'refrigerante': 'Temos diversas opções, para continuar o pedido acesse o link meusite.com/produtos/refrigerantes',
        'cerveja': 'Quer uma gelada? Confira as nossas cervejas no link: meusite.com/produtos/cervejas',
        'suco': 'Escolha entre nossos sucos frescos no link: meusite.com/produtos/sucos',
        'gelo': 'Precisa de gelo? Faça seu pedido aqui: meusite.com/produtos/gelo',
        'agua': 'Água mineral disponível, peça no link: meusite.com/produtos/agua'
    },
    'consultar': {
        'refrigerante': 'Temos várias marcas de refrigerantes como Vedete, Pepsi, Fanta, entre outras.',
        'cerveja': 'Nossas opções de cerveja incluem marcas como Heineken, Budweiser, Skol e outras.',
        'suco': 'Oferecemos sucos de frutas variados, como laranja, uva e manga.',
        'gelo': 'Vendemos pacotes de gelo de 1kg e 5kg.',
        'agua': 'Água mineral disponível em garrafas de 500ml, 1L e 5L.'
    },
    'preço': {
        'refrigerante': 'Os preços dos refrigerantes variam entre R$ 5,00 e R$ 10,00, dependendo da marca e do tamanho.',
        'cerveja': 'Cervejas a partir de R$ 7,00, confira as opções no site.',
        'suco': 'Os sucos estão na faixa de R$ 6,00 a R$ 12,00, conforme a marca e o tamanho.',
        'gelo': 'Pacotes de gelo disponíveis por R$ 4,00 (1kg) e R$ 15,00 (5kg).',
        'agua': 'Água mineral a partir de R$ 2,50 para garrafas de 500ml.'
    }
}

def getEntity(tokens):
  for token in tokens:
    for entity, values in entities.items():
      if token in values:
        return entity

def getIntention(tokens):
  for token in tokens:
    for intention, values in intentions.items():
      if token in values:
        return intention

def chatbot():
  print('Bem vindo a Bebidas S/A. Como posso ajudar?')
  while True:
    answer = input('Digite a pergunta ou sair: ')
    if answer == 'sair':
      break
    tokens = preprocessing(answer)
    entity = getEntity(tokens)
    intention = getIntention(tokens)

    if entity and intention:
      print(responses[intention][entity])
    elif intention:
      print(f'Entendi! Você quer falar sobre {intention}. Explique melhor.')
    elif entity:
      print(f'Entendi! Você quer falar sobre {entity}. Explique melhor.')
    else:
      print('Desculpe, não entendi a pergunta.')

    # print(tokens)
    # print(f'Entidade: {entity}')
    # print(f'Intenção: {intention}')

chatbot()