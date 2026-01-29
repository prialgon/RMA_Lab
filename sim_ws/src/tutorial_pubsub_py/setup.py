from setuptools import find_packages, setup

package_name = "tutorial_pubsub_py"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Coolgolf1",
    maintainer_email="coolgolfmaster@gmail.com",
    description="Examples of minimal publisher/subscriber using rclpy",
    license="Apache License 2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "talker = tutorial_pubsub_py.publisher_member_function:main",
            "listener = tutorial_pubsub_py.subscriber_member_function:main",
        ],
    },
)
