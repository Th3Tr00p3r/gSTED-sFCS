# gSTED-sFCS

Python-based software for our optical measurement system.

## How this project works:

#### <u>GUI</u>

1. .ui files are created graphically via ''Qt Designer', then are compiled into .py files (e.g. `mainwindow.ui` compiles into `Ui__mainwindow.py`) which are just the translation of the graphical layout into PyQt5 code.

2. These files are then imported into the `mainwindow.py` module, where the signals/slots (similar to callbacks) are coded as methods for the classes that are named after the windows (`MainWindow`, `SettingsWindow`, etc.).

3. Finally, when running the project (`gSTEDsFCS.py`), a `MainWindow()` object is instantiated and the window is shown on screen.

#### <u>Physical Device Control</u>

1. the `drivers` package contains an `__init__.py` file which imports the necessary drivers from existing packages (pyVISA, nidaqmx, instrumental etc.), which provide the interface to the physical devices.

2. `drivers` is imported into the implementation modules which communicate with the devices.

#### <u>Implemetation</u>

1. ## What's here/Noteworthy files:
- `./gSTEDsFCS.e4p` - 'eric IDE' project file, which nicely ties everything together.

- `./ui/mainwindow.py` - GUI module.

- `./implementation/implementation.py` - main implementation module.

- `./implementation/constants.py` - constants used across the project.

- `./drivers` - package for physical devices interfaces (used only in implementation modules).

- `./settings/default_settings.csv` - default setting file, initial values for all GUI forms.
