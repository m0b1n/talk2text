[app]
title = Talk2Text
project_dir = .
input_file = packaging/main.py
exec_directory = deployment
icon = packaging/io.github.m0b1n.Talk2Text.svg

[python]
python_path = /home/m0b1n/Documents/dev/talk2text/.venv/bin/python3
packages = nuitka==2.8.4,ordered_set,zstandard,patchelf

[qt]
qml_files = 
plugins = multimedia,networkinformation,platforminputcontexts
excluded_qml_plugins = QtQuick,QtQuick3D,QtCharts,QtWebEngine,QtTest,QtSensors
modules = Concurrent,Core,DBus,Gui,Multimedia,Network,Widgets

[nuitka]
mode = standalone
extra_args = --quiet --noinclude-qt-translations

