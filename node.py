class node():
    def __init__(self,name):
        self.connections_in = []
        self.connections_out = []
        self.name = name

    def add_connection(self,other):
        self.connections_out.append(other)
        other.connections_in.append(self)
