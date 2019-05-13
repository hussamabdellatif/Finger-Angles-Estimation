import sys
import RPi.GPIO as G
import time
import socket
def angleToDuty(angle):
	return 0.1071 * angle + 5.3571

frequency = 100
pin1 = 12
pin2 = 33
pin3 = 35

G.setmode(G.BOARD)

#setup pins 12,33,35
G.setup(pin1,G.OUT)
G.setup(pin2,G.OUT)
G.setup(pin3,G.OUT)

# Setup PWM connections
p1 = G.PWM(pin1, frequency)
p2 = G.PWM(pin2, frequency)
p3 = G.PWM(pin3, frequency)

#intialize angles
angles = [135,135,135]

p1.start(angleToDuty(angles[0]))
p2.start(angleToDuty(angles[1]))
p3.start(angleToDuty(angles[2]))

#setup server
ipaddr = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]
port = 40123
size = 64

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((ipaddr,port))
server.listen(5)
print('SERVER READY.......\n\n')
while True:
    conn, addr = server.accept()
    print('connection estabilished')
    while True:
        try:
            data = conn.recv(size)
            data = str(data)
            data_list = data.split(',')
            angles = [int(i) for i in data_list]
            angles_valid = False
            terminate = False
            for angle in angles:
                if angle >=20 and angle <=135:
                    angles_valid = True
                elif angle <= 0:
                    terminate = True
                else:
                    angles_valid = False
            if terminate:
                conn.close()
                server.close()
                break
            if angles_valid:
                print(angles)
                print('\n')
                p1.ChangeDutyCycle(angleToDuty(angles[0]))
                p2.ChangeDutyCycle(angleToDuty(angles[1]))
                p3.ChangeDutyCycle(angleToDuty(angles[2]))
            conn.send('I am server')

        except:
            print('ERROR: TERMINATING\n')
            conn.close()
            server.close()
            break
    break

G.cleanup()




