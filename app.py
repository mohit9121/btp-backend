from flask import Flask

app = Flask(__name__) 

@app.route('/')
def hello_world():
    return 'Backend is running. BTP of Mohit Anirudh and Aman IIT Ropar'

@app.route('/test')
def hello_world2():
    return 'Backend is running testing '

if __name__ == '__main__':
    app.run()
