from minigrid.manual_control import ManualControl

from stochastic_envs.teleport import Teleport5by5


def main():
    env = Teleport5by5(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
