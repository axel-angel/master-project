import sys
from SimpleHTTPServer import SimpleHTTPRequestHandler
import BaseHTTPServer

if __name__ == "__main__":
    HandlerClass=SimpleHTTPRequestHandler
    ServerClass=BaseHTTPServer.HTTPServer

    protocol = "HTTP/1.0"
    host, port = sys.argv[1].split(':')
    port = int(port)

    server_address = (host, port)

    HandlerClass.protocol_version = protocol
    httpd = ServerClass(server_address, HandlerClass)

    sa = httpd.socket.getsockname()
    print "Serving HTTP on", sa[0], "port", sa[1], "..."
    httpd.serve_forever()
