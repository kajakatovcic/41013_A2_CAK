from ir_support_extra_robots.robots import OmronTM5700
import swift

env = swift.Swift()
env.launch(realtime=True)

omron = OmronTM5700()
omron.add_to_env(env)
omron.test()