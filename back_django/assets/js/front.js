
			  console.log(tooltip.body)
			 
			  let dataToDisplay;
			 
			  if (tooltip.body.includes("deep")) {
				dataToDisplay = {
					"title": "Deep Q-Learning",
					// "AVG Rewards": parseFloat("{{ q_learning_best_result.avg_rewards }}".replace(/,/g, '.')),
					// "AVG Steps": parseFloat("{{ q_learning_best_result.avg_steps }}".replace(/,/g, '.'))
					// "Duration": parseFloat("{{ q_learning_best_result.duration }}".replace(/,/g, '.'))
				  }
			  } else if (tooltip.body.includes("learning")) {
				dataToDisplay = {
					"title": "Q-Learning",
					"AVG Rewards": parseFloat("{{ q_learning_best_result.avg_rewards }}".replace(/,/g, '.')),
					"AVG Steps": parseFloat("{{ q_learning_best_result.avg_steps }}".replace(/,/g, '.'))
					"Duration": parseFloat("{{ q_learning_best_result.duration }}".replace(/,/g, '.'))
				}
			} else if (tooltip.body.includes("sarsa")) {
				dataToDisplay = {
					"title": "Sarsa",
					// "AVG Rewards": parseFloat("{{ q_learning_best_result.avg_rewards }}".replace(/,/g, '.')),
					// "AVG Steps": parseFloat("{{ q_learning_best_result.avg_steps }}".replace(/,/g, '.'))
					 //"Duration": parseFloat("{{ q_learning_best_result.duration }}".replace(/,/g, '.'))
				}
			} else if (tooltip.body.includes("user")) {
				dataToDisplay = {
					"title": "User Parameters",
					"AVG Rewards": parseFloat("{{ algo_user_result.avg_rewards }}".replace(/,/g, '.')),
					"AVG Steps": parseFloat("{{ algo_user_result.avg_steps }}".replace(/,/g, '.'))
					"Duration": parseFloat("{{ algo_user_result.duration }}".replace(/,/g, '.'))
				}
			}