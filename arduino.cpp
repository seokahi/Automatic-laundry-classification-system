#include <Servo.h>

Servo myservo;

int stepPin[] = {8, 9, 10, 11};

void aStep(int s)
{
    switch (s)
    {
    case 0:
        digitalWrite(stepPin[0], LOW);
        digitalWrite(stepPin[1], HIGH);
        digitalWrite(stepPin[2], HIGH);
        digitalWrite(stepPin[3], LOW);
        break;
    case 1:
        digitalWrite(stepPin[0], LOW);
        digitalWrite(stepPin[1], HIGH);
        digitalWrite(stepPin[2], LOW);
        digitalWrite(stepPin[3], HIGH);
        break;
    case 2:
        digitalWrite(stepPin[0], HIGH);
        digitalWrite(stepPin[1], LOW);
        digitalWrite(stepPin[2], LOW);
        digitalWrite(stepPin[3], HIGH);
        break;
    case 3:
        digitalWrite(stepPin[0], HIGH);
        digitalWrite(stepPin[1], LOW);
        digitalWrite(stepPin[2], HIGH);
        digitalWrite(stepPin[3], LOW);
        break;
    default:
        break;
    }
}

void toStart()
{
    while (true)
    {
        if (digitalRead(2))
            break;
        else
        {
            doSteps(1, 8, 2);
        }
    }
}
void toCenter()
{
    if (digitalRead(2))
    {
        while (true)
        {
            if (digitalRead(3))
                break;
            else
            {
                doSteps(0, 8, 2);
            }
        }
    }
    if (digitalRead(4))
    {
        while (true)
        {
            if (digitalRead(3))
                break;
            else
            {
                doSteps(1, 8, 2);
            }
        }
    }
    DropTheCloth();
}
void toEnd()
{
    while (true)
    {
        if (digitalRead(4))
            break;
        else
        {
            doSteps(0, 8, 2);
        }
    }
}

void loop()
{
    char button_left = digitalRead(2);
    char button_center = digitalRead(3);
    char button_right = digitalRead(4);

    int val_step = analogRead(A0);
    int val_servo = analogRead(A1);

    if (Serial.available() > 0)
    {
        data = Serial.read();

        if (data == '1')
        {
            toStart();
        }
        else if (data == '2')
        {
            toStart();
        }
        else if (data == '3')
        {
            toEnd();
        }
        else if (data == '4')
        {
            toEnd();
        }
    }
}