import json
from pathlib import Path
import uuid
from typing import Any, Callable, Set

# Create a function to submit a support ticket
def submit_support_ticket(email_address: str, description: str) -> str:
     script_dir = Path(__file__).parent  # Get the directory of the script
     ticket_number = str(uuid.uuid4()).replace('-', '')[:6]
     file_name = f"ticket-{ticket_number}.txt"
     file_path = script_dir / file_name
     text = f"Support ticket: {ticket_number}\nSubmitted by: {email_address}\nDescription:\n{description}"
     file_path.write_text(text)
    
     message_json = json.dumps({"message": f"Support ticket {ticket_number} submitted. The ticket file is saved as {file_name}"})
     return message_json




import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Replace these with your actual SMTP credentials
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email_id"
SENDER_PASSWORD = "your_password"  # Use App Password for Gmail

def send_email(to_email: str, subject: str, body: str) -> bool:
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())

        return True
    except Exception as e:
        print("Error sending email:", str(e))
        return False


def escalate_ticket_with_email(issue: str, ticket_id: str) -> dict:
    subject = "Escalation: Unresolved IT Issue"
    body = f"""
    Hello IT Support Team,

    The following issue reported by a user could not be resolved by the AI Assistant:
    Ticket Id: {ticket_id}
    "{issue}"

    Please investigate and take further action.

    Regards,
    AI Notification Agent
    """

    success = send_email(to_email="sandeshhase15@gmail.com", subject=subject, body=body)
    return {"content": "📧 Email sent to IT support." if success else "⚠️ Failed to send email."}


# Define a set of callable functions
user_functions: Set[Callable[..., Any]] = {
     submit_support_ticket, escalate_ticket_with_email
 }

# # Define a set of callable functions
# user_functions: Set[Callable[..., Any]] = {
#      submit_support_ticket
#  }