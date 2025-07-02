from setuptools import find_packages, setup

package_name = 'poly_approx_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lar95',
    maintainer_email='mathijs.svrc@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "curve_writer = poly_approx_control.curve_writer:main",
            "data_aquisition = poly_approx_control.data_aquisition:main",
            "beta_computer = poly_approx_control.beta_computer_6dof:main",
            "A_init_computer = poly_approx_control.A_init_computer:main",
            "cmd_from_jac = poly_approx_control.cmd_from_jac:main",
            "A_tester = poly_approx_control.A_tester:main",
            "beta_tester = poly_approx_control.beta_tester:main",
            "broyden_controller = poly_approx_control.broyden_controller:main",
            "beta_computer_node = poly_approx_control.beta_computer_node:main",
            "manual_curve_writer = poly_approx_control.manual_curve_writer:main",
            "ds_dr_pub = poly_approx_control.ds_dr_pub:main",
            "broyden_controller_senza_A_init = poly_approx_control.broyden_controller_senza_A_init:main",
            "broyden_controller_w_A_init = poly_approx_control.broyden_controller_w_A_init:main",
            "A_tester_real = poly_approx_control.A_tester_real:main",
            "poly_approx_controller = poly_approx_control.poly_approx_controller:main",

        ],
    },
)
