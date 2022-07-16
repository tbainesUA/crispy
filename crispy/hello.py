import numpy as np

class Person:
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender
    
    def greetings(self):
        # print(f'{self.name} say sew his butthole up!')
        print('%s say tattoos his butthole' % self.name)
        

def do_something():
    # some commands to run
    pass

def add():
    a = 5
    b = 2
    a_plus_b = a + b
    print(a, ' + ', b, ' = ', a_plus_b)

def main():
    add()
    person = Person(name='IZZY', age='Unknown', gender='homophobe')
    
    if person.name !=  'XYZ':
        print(person.name)
        person.greetings()
    else:
        print('No person name XYZ')
    print('Hello World')
      

if __name__ == '__main__':
    main()
    
