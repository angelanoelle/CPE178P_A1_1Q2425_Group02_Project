class User:
    def __init__(self, username, password, fullname):
        self.username = username
        self.password = password
        self.fullname = fullname
    def set_username(self,username):
        self.username = username
        
    def set_password(self,password):
        self.password = password
    def set_fullname(self,fullname):
        self.fullname = fullname
    def get_username(self,username):
        return self.username
    def get_password(self,password):
        return self.get_password
    
    def get_fullname(self,fullname):
        return self.fullname

