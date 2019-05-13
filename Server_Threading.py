# This is an example of Threading
# Goal of this Work:
	# (a) - To demonstrate concurrent process operation for server and motor control
#https://www.tutorialspoint.com/python3/python_multithreading.htm
#!/usr/bin/python3

import threading
import time
import socket
import sys
import datetime
import re
import RPi.GPIO as GPIO



# Remember we want simply a main and a signle server thread
# May need to make it a Daemon Thread Later on

# Later add functionality for angles in this
class motors (threading.Thread):
	def __init__(self, frequency):
		threading.Thread.__init__(self)
		self.pinLocation = [12, 33, 35]
		self.frequency = frequency
		
	def run(self):
		GPIO.setmode(GPIO.BOARD)
		GPIO.setup(self.pinLocation[0],GPIO.OUT)
		GPIO.setup(self.pinLocation[1],GPIO.OUT)
		GPIO.setup(self.pinLocation[2],GPIO.OUT)
	  
		p1 = GPIO.PWM(self.pinLocation[0],self.frequency)
		p2 = GPIO.PWM(self.pinLocation[1],self.frequency)
		p3 = GPIO.PWM(self.pinLocation[2],self.frequency)
	  
		global motorAngles	  
		p1.start(angleToDuty(motorAngles[0]))
		p2.start(angleToDuty(angleCorrection(motorAngles[1])))
		p3.start(angleToDuty(angleCorrection(motorAngles[2])))
		
		while True:
			try:
				if(motorAngles == [666,666,666]):
					p1.stop()
					p2.stop()
					p3.stop()
					GPIO.cleanup()
				if(motorAngles[0]>=20 and motorAngles[0]<=135):					
					p1.ChangeDutyCycle(angleToDuty(motorAngles[0]))
				if(motorAngles[1]>=20 and motorAngles[1]<=135):	
					p2.ChangeDutyCycle(angleToDuty(angleCorrection(motorAngles[1])))
				if(motorAngles[2]>=20 and motorAngles[2]<=135):	
					p3.ChangeDutyCycle(angleToDuty(angleCorrection(motorAngles[2])))	
				
			except KeyboardInterrupt:
				p1.stop()
				p2.stop()
				p3.stop()
				GPIO.cleanup()	  
	  
def angleToDuty(angle):
	return 0.0536 * angle + 2.6786

def angleCorrection(angle): #Correction for the top two motors
	i = 135 - angle
	return 20 + int(i * 0.6521739)

class myServer (threading.Thread):
	def __init__(self, host, port, size, backlog): #threadID, name, counter):
		threading.Thread.__init__(self)
		self.host = host
		self.daemon=True
		self.port = port
		self.size = size
		self.backlog = backlog
		self.server_address = (host, port)
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.bind(self.server_address)

	def run(self):
		print >>sys.stderr, 'Starting up on %s port %s...' % self.server_address	
		self.sock.listen(self.backlog)
		while True:
			#flag = ''
			print >>sys.stderr, 'Waiting for a connection...'
			connection, client_address = self.sock.accept()
			
			try:
				print >>sys.stderr, 'Connection from', client_address, 'at', datetime.datetime.now()
				while True:
					try:
						data = connection.recv(self.size)
						global motorAngles
						
						h = str(data)
						matches = re.findall("(\d+)", h)
						t = 0
						for i in matches:
                                                        if t < len(motorAngles):
                                                                motorAngles[t] = int(i)
                                                                t+=1
						print(motorAngles)
						print('\n')
						if data:
                                                        data = data
						else:
							print >>sys.stderr, 'No more data from', client_address
							break
						if(motorAngles == [666,666,666]):
							connection.close()
							#sys.exit(0)						
					except (KeyboardInterrupt, SystemExit):
						connection.close()
						self.sock.close()
						sys.exit()
			finally:
				connection.close()
				
##################################################################################################
	# INITIALIZATIONS #
##################################################################################################

motorAngles = [135,135,135]
# ALL MOTORS NEED TO BE PUT ON FINGER AT 45 DEGREES WHEN YOU PUT THEM IN


Host = '67.20.211.228'
Port = 40323
Backlog = 5
Size = 64

frequency = 100
threadServer = myServer(Host,Port,Size,Backlog)
threadMotor = motors(frequency)

threadServer.start()
threadMotor.start()

threadMotor.join()
threadServer.join()
