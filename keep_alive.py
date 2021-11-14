# ****************************************
# Universidad del Valle de Guatemala
# Programacion de microprocesadores
#
# Author: AllAwsome497
# Modifier: Diego Cordova
# ****************************************

# Libraries
from flask import Flask
import random
import requests
import time
from multiprocessing import Process

# ----------------------- Site ----------------------
app = Flask('')
port1 = random.randint(2000,9000)
@app.route('/')

def home():
	return 'Im in!'

def run():
  	app.run(
		host='0.0.0.0',
		port=port1
	)

server_thread:Process = Process(target=run)

def start():
	'''
	Creates and starts new thread that runs the function run.
	'''
	server_thread.start()
	time.sleep(2)

def keep_alive():
	URL = 'http://0.0.0.0:' + str(port1) + '/'
	requests.get(url = URL)

def end():
	server_thread.terminate()
	server_thread.join()
	print('\n------------------------------------------------------------------------')
	print('-> Server Terminated')
	print('\n------------------------------------------------------------------------')

