# sFCS-gSTED

GUI for controlling our optical measurement system.

## How This project works:

#### <u>GUI</u>

1. .ui files are created graphically via ''Qt Designer', then are compiled into .py files (e.g. mainwindow.ui ----> Ui__mainwindow.py) which are just the translation of the graphical layout into PyQt5 code.

2. These files are then imported into the mainwindow.py module, where the signals/slots (similar to callbacks) are coded as methods for the classes that are named after the windows (MainWindow, SettingsWindow, etc.).

3. Finally, when running the project (gSTEDsFCS.py), a MainWindow() object is instantiated and the window is shown on screen.

#### <u>Physical Device Control</u>

## What's here:

- gSTEDsFCS.e4p - 'eric IDE' project file, which nicely ties everything together

- mainwindow.py - module for signals & slots using PyQt5, containing GUI window classes
