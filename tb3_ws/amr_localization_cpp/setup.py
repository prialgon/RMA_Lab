from setuptools import setup

package_name = "amr_localization_cpp"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/lab03.launch.py"]),
        ("share/" + package_name + "/maps", ["maps/lab03.json"]),
        ("share/" + package_name + "/launch", ["launch/lab04.launch.py"]),
        ("share/" + package_name + "/maps", ["maps/project.json"]),
    ],
    zip_safe=False,
)