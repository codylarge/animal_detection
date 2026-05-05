import os
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
from pyftpdlib.authorizers import DummyAuthorizer

FTP_HOST = "0.0.0.0"   # listen on all network interfaces
FTP_PORT = 21
FTP_USER = "deckcamera"
FTP_PASS = "2534"
FTP_DIR = os.path.abspath("ftp_uploads")

def main():
    authorizer = DummyAuthorizer()
    authorizer.add_user(FTP_USER, FTP_PASS, FTP_DIR, perm="elradfmwMT")

    handler = FTPHandler
    handler.authorizer = authorizer
    handler.passive_ports = range(60000, 60100)  # passive mode port range
    handler.masquerade_address = "192.168.1.XXX"  # your machine's local IP

    server = FTPServer((FTP_HOST, FTP_PORT), handler)
    print(f"FTP server running on port {FTP_PORT}, watching {FTP_DIR}")
    server.serve_forever()

if __name__ == "__main__":
    main()