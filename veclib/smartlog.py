from veclib.utils import *
from twilio.rest import Client


class SmartLog:

    time_stamp = None
    pid = 0
    def __init__(self):
        name = ''
        start_time=time.time()
        output_folder = '../output/'+ name


    def format_name(self):
        self.name = "[{0}]-[]"




def send_log(message):
	"""
	send log string as a message to the phone number. need to register twilio before use
	Args:
		message: message to be sent
	Returns:
		no return
	"""
	account_sid = "ACa6120ed7fd410c4b56030efa0ba9d6a2" # Your Account SID from www.twilio.com/console
	auth_token	= "8e8f6a573d9b868b8564ee3989996f01"	# Your Auth Token from www.twilio.com/console
	client = Client(account_sid, auth_token)
	message = client.messages.create(body=message,
		to="+15197213077",		# Replace with your phone number
		from_="+12268871473") # Replace with your Twilio number