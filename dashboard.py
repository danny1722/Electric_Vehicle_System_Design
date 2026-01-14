import globals
from flask import Flask, request, render_template
from main import main

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])


def submit():
    if request.method == 'POST':
        print(request.form)
        globals.safety_factor = float(request.form['safety_factor'])
        globals.mass = int(request.form['chassis_mass'])
        globals.Cd = float(request.form['Cd'])
        globals.Af = float(request.form['Af'])
        globals.Cr = float(request.form['Cr'])
        globals.regen_eff = float(request.form['regen_eff'])
        globals.motor_eff = float(request.form['motor_eff'])
        globals.round_trips = int(request.form['round_trips'])
        if 'using_pantograph' in request.form:
          globals.using_pantograph = True
        else:
          globals.using_pantograph = False
            
        main(debug = False)
    batteryMass = int(globals.battery_capacity * 1000 / globals.Li_ion_energy_density)
    usePantograph=''
    if globals.using_pantograph:
      usePantograph = 'checked'
    return render_template('main.html', 
                           mass = globals.mass + batteryMass,
                           max_speed=int(globals.max_speed*3.6),
                           batteryCapacityStr=int(globals.battery_capacity), 
                           batteryMass = batteryMass,
                           safety_factor = globals.safety_factor,
                           chassis_mass=globals.mass,
                           Cd = globals.Cd,
                           Af = globals.Af,
                           Cr = globals.Cr,
                           regen_eff = globals.regen_eff,
                           motor_eff = globals.motor_eff,
                           round_trips = globals.round_trips,
                           using_pantograph = usePantograph)

if __name__ == '__main__':
    main(debug = False)
    app.run(debug=True)

