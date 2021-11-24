# ------------------------------------------------------
# Universidad del Valle de Guatemala
# Programacion de Microprocesadores
#
# datos.py:
# - Lectura de datos de sensor 
#   y escritura de archivo Json
# 
# Autor: Diego Cordova - 20212
# ------------------------------------------------------

# librerias importadas
import adafruit_dht
import board
import json
import time
import datetime as dt
import keep_alive

# ------------------------ Funcion de guardado en Json ------------------------

def GuardarJSON(data):
    newJson = json.dumps(data, indent=4)

    with open('Datos_Sensor.json', 'w') as outfile:
        outfile.write(newJson)

    print('Archivo Json escrito con exito!!\n')

# ------------------------ Lectura de sensor de humedad DHT11 ------------------------

sensorData = {} # Diccionario para agregar datos
sensorData["DHT11"] = [] #Sensor DHT11 
dht = adafruit_dht.DHT11(board.D4) # conexion con sensor
keep_alive.start() # Empieza el servidor

# Ciclo principal
for i in range(0, 15000):

    while True:

        try:
            temperature = dht.temperature
            humidity = dht.humidity
            deltaTime = dt.datetime.now().strftime('%H:%M:%S')

            if humidity is None or temperature is None:
                raise RuntimeError

            sensorData["DHT11"].append({
                'temp' : str(temperature),
                'humedad' : str(humidity),
                'time' : deltaTime
            })

            print('\n------------------------------------------------------------------------')
            print(f"Temp: {temperature} ºC \t Humidity: {humidity}% \t Time: {deltaTime}")
            break

        except RuntimeError as e:
            # Reading doesn't always work! Just print error and we'll try again
            # print("Reading from DHT failure: ", e.args)
            pass

    GuardarJSON(sensorData) # Guarda Json
    keep_alive.keep_alive() # Mantiene activo el servidor
    time.sleep(15)           # Espera 15 segs

keep_alive.end()            # Termina el servidor

