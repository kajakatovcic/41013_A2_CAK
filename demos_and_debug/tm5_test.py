# This is a simple test of the Omron TM5700 model in Swift.
# The model is created and added to the Swift environment and the test()
# methd is called to perform a simple motion test.
# This is a temporary test file and will be removed once the Omron TM5700
# model is fully integrated into the main codebase.
from ir_support_extra_robots.robots import OmronTM5700
import swift

env = swift.Swift()
env.launch(realtime=True)

omron = OmronTM5700()
omron.add_to_env(env)
omron.test()