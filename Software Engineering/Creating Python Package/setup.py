import setuptools

setuptools.setup(
    name="travelplanner",
    version="0.0.1",
    author="Jack Weeks",
    author_email="jack.weeks.16@ucl.ac.uk",
    description="Plan your appropriate modes of transport for your journey",
    packages=setuptools.find_packages(exclude=['*tests']),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'pyyaml', 'matplotlib', 'pytest', 'argparse'],
    entry_points={
        'console_scripts': [
            'bussimula = travelplanner.command:travel_planner_input'
        ]})
