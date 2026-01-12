from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])


def submit():
    if request.method == 'POST':
        batteryCapacity = request.form['battery-capacity']
        return render_template('main.html', batteryCapacity=batteryCapacity)
    return render_template('main.html', batteryCapacity=batteryCapacity)

if __name__ == '__main__':
    batteryCapacity = 0
    app.run(debug=True)

