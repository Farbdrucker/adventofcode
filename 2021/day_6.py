# https://adventofcode.com/2021/day/6
from dataclasses import dataclass
from typing import List

creation_rate: int = 7


@dataclass
class LanternFish:
    state: int

    def __call__(self):
        if self.state == 0:
            self.state = 6
            return self, LanternFish(8)
        else:
            self.state -= 1
            return self, None

    def __repr__(self):
        return str(self.state)


@dataclass
class Swarm:
    fish: List[LanternFish]

    def __call__(self):
        fish, new = zip(*[fish() for fish in self.fish])
        self.fish = list(fish) + list(filter(lambda x: x is not None, new))

    def __len__(self):
        return len(self.fish)

    def __repr__(self):
        def join(fish) -> str:
            return ",".join([str(f) for f in fish])

        if len(self) < 16:
            return join(self.fish)

        else:
            return join(self.fish[:15]) + ",...," + join(self.fish[-2:])


def breeding(*state: int):
    return [LanternFish(s) for s in state]


start = [3, 4, 3, 1, 2]
max_days = 80
swarm = Swarm(breeding(*start))

for day in range(max_days):
    swarm()

    print(f"[{len(swarm):4d}]After {day+1:3d} days: {swarm}")
