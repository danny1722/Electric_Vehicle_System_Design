from flask import Flask, render_template, request 
app = Flask(__name__) 
@app.route('/form', methods=['GET', 'POST']) 
def form_view(): 
  if request.method == 'POST': # Process form data 
    pass 
  return render_template('form.html')
