import subprocess

program_list = [
                "./Analysis_hcp.py",
                "./Analysis_abide.py",
                "./Analysis_IXI.py",
                "./Analysis_bcw.py"
                ]

for program in program_list:
    subprocess.call(['python3', program])
