from flask import Flask,request,render_template,jsonify#Conjunto de classes reponsáveis por controlar requisições web
from flask_cors import CORS# controle de acesso

app = Flask(__name__)#cria o objeto Flask
CORS(app, origins='*')#libero acesso a qualquer dispositivo 

@app.route('/')
def noite():
    return jsonify({"reposta":"Boa noite"})

@app.route('/soma/<int:valor1>/<int:valor2>')#soma 2 valores
#soma 2 numeros 
def soma(valor1,valor2):
    return jsonify({"resultado": valor1 + valor2})
    
@app.route('/subtracao/<int:valor1>/<int:valor2>',methods=['PUT'])#subtrai 2 valores
def subtracao(valor1,valor2):
    return jsonify({"resultado": valor1 - valor2})

if __name__ == '__main__':
    app.run(debug=True)




