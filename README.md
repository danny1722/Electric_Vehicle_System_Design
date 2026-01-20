## How to run

- To use the UI, run dashboard.py and navigate to the link provided in the terminal (default http://127.0.0.1:5000)

## Inputs

- Varius inputs are available in the ui that can change the train's performance:
  - Safety factor: How much battery charge should be left at the end of the day
  - Pantograph: Whether or not to use pantographs to charge and power the train when overheadlines are availalbe
  - Chassis: Several characteristics of the train's chassis. These mainly impact the amount of resistance the train faces and how much power it needs to move itself
  - Regenerative breaking efficiency: How much of the energy used can be recovered through breaking
  - Motor and drivetrain efficiency: The ratio of mechanical power delivered at the wheels to the electrical power drawn from the battery.
  - Number of round trips: How often the train will travel back and forth before needing to fully charge
