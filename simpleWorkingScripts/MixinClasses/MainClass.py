from simpleWorkingScripts.MixinClasses.SubClass1 import SubClass1
from simpleWorkingScripts.MixinClasses.SubClass2 import SubClass2


class MainClass(SubClass1, SubClass2):

    def __init__(self):
        self.arg1 = 1
        self.arg2 = 2
        SubClass1.__init__(self)
        SubClass2.__init__(self)


mc = MainClass()

mc.print1()
mc.print2()
mc.p1()
mc.p2()