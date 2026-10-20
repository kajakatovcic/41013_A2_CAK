# This is a simple test of the UR3e model in Swift.
# The model is created and added to the Swift environment and the test()
# methd is called to perform a simple motion test.
# This is a temporary test file and will be removed once the UR3e
# model is fully integrated into the main codebase.
from ir_support import UR3e
import swift

env = swift.Swift()
env.launch(realtime=True)

ur3e = UR3e()
ur3e.add_to_env(env)
ur3e.test()