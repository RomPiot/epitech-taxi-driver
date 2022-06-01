from django import forms


# nb_episodes = int(input("Episode Iterations : (default = 10 000)") or 10000)
#   epsilon_rate = float(input("Epsilon Rate : (default = 1.0)") or 1.0)
#   epsilon_min = float(input("Epsilon Min :(default = 0.01)") or 0.01)
#   epsilon_max = float(input("Epsilon Max : (default = 1.0)") or 1.0)
#    learning_rate= float(input("Learning Rate : (default = 0.8)") or 0.8)
#   reward_discount_rate = float(input("Reward Discount Rate : (default = 0.786)") or 0.786)
#   decay_rate = float(input("Decay Rate : (default = 0.07)") or 0.07)


class UserInputForm(forms.Form):
    nb_episodes = forms.IntegerField()
    epsilon_rate = forms.FloatField()
    epsilon_min = forms.FloatField()
    epsilon_max = forms.FloatField()
    learning_rate = forms.FloatField()
    reward_discount_rate = forms.FloatField()
    decay_rate = forms.FloatField()
