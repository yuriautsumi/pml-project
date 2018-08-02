function bool = eval_error(prediction, ground_truth, threshold)
% Returns boolean: true if abs(prediction-ground_truth) > threshold 
if abs(prediction-ground_truth) > threshold 
    bool = true; 
else
    bool = false; 
end 
end 