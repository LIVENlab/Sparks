from dataclasses import asdict

from pydantic.dataclasses import dataclass


class AX:

    @classmethod
    def a(clazz):
        print()
        return clazz


AX.a()
x = AX()
x.a()

@dataclass
class DC:
    a: int

asdict(DC(2))
