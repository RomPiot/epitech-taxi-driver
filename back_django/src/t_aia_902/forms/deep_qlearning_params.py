from django import forms


CHOICES = [
    (100000, "100 000"),
    (200000, "200 000"),
    (300000, "300 000"),
    (400000, "400 000"),
    (500000, "500 000"),
    (600000, "600 000"),
    (700000, "700 000"),
    (800000, "800 000"),
    (900000, "900 000"),
    (1000000, "1 000 000"),
]


class DeepQLearningParamsForm(forms.Form):
    nb_episodes = forms.ChoiceField(choices=CHOICES, help_text="default = 1 000 000", initial=1000000)
