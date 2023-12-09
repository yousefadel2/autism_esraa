# This file is executed on every boot (including wake-boot from deepsleep)
#import esp
#esp.osdebug(None)
#import webrepl
#webrepl.start()
import network
from umqtt.simple import MQTTClient
import camera
import time
import socket
import machine

# Connect to Wi-Fi

# Initialize the camera
adc = machine.ADC(2)
heart_rate_value = adc.read()
sensor_min = 0
sensor_max = 4095
# Raw ADC value
raw_value = heart_rate_value
max_heart_rate=100
min_heart_rate=60
# Scale the raw value to obtain a heart rate value
heart_rate = (raw_value - sensor_min) / (sensor_max - sensor_min) * (max_heart_rate - min_heart_rate) + min_heart_rate

print("heart rate = ",heart_rate)

camera.init(0, format=camera.JPEG)
camera.framesize(camera.FRAME_SVGA  )
camera.brightness(2)
# -2,2 (default 0). 2 brightness
camera.saturation(-2)
#camera.speffect(camera.EFFECT_NONE  )
#camera.whitebalance(camera.WB_NONE  )

camera.contrast(-2)
#-2,2 (default 0). 2 highcontrast

# quality
camera.quality(10)
# 10-63 lower number means higher quality
from machine import Pin
import time

# GPIO pin connected to the LED
led_pin = Pin(4, Pin.OUT)  # Replace '2' with the actual GPIO pin number

def turn_on_led():
    led_pin.on()

def turn_off_led():
    led_pin.off()
turn_on_led()
  # Keep the LED on for 2 seconds

photo = camera.capture()
time.sleep(2)
turn_off_led()
camera.deinit()
# Connect to MQTT server
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect('jooo', 'hhhh1111')

# Wait for connection
while not wlan.isconnected():
    pass

client = MQTTClient("heartbeat", "192.168.77.179")
client2 = MQTTClient("espcam", "192.168.77.179")

client.connect()
client2.connect()
i=0
while i<1:
    client.publish("heartbeat", str(heart_rate))
    client2.publish("photo", photo)    

    i=i+1


