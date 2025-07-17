import os
import sys
import numpy as np
import pandas as pd
import logging
from collections import OrderedDict
from dotenv import load_dotenv
from openai import OpenAI
from linear_boost_model import LinearBoostModel
from config_parse import load_toml_config
from utility_fun import capture_model_state, setup_logging, LoggerWrapper, load_checkpoint, save_checkpoint, plot_graph, save_results
from data_process_fun import preprocess_data, compute_metrics, compute_accuracy_in_bounds
from string_manipulation import build_feature_prompt, parse_feature_response, generate_readable_feature_name


def load_config():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the absolute path to the config file
    config_path = os.path.join(script_dir, '..', '..', 'config', 'modular.toml')

    config = load_toml_config(config_path)

    return config

def load_data(config):
    # Load configuration
    local_path = config['info']['local_path']
    features = config['info']['numerical_features']

    # Load data
    train_path = os.path.join(local_path, config['info']['train_set'])
    val_path = os.path.join(local_path, config['info']['val_set'])
    test_path = os.path.join(local_path, config['info']['test_set'])
    diff_path = os.path.join(local_path, config['info']['diff_set'])

    # Load data
    df_train_raw = pd.read_csv(train_path)
    df_val_raw = pd.read_csv(val_path)
    df_test_raw = pd.read_csv(test_path)
    df_diff_raw = pd.read_csv(diff_path)
    df_diff_raw['q_gonality_bounds']=df_diff_raw['q_gonality_bounds'].apply(lambda x: eval(x) if isinstance(x, str) else x)


    # Filter to only features + target column for extraction
    target_col = 'q_gonality'
    # Extract targets
    target_train = df_train_raw[target_col]
    target_val = df_val_raw[target_col]
    target_test = df_test_raw[target_col]
    target_diff = df_diff_raw['q_gonality_bounds']

    df_train_raw = df_train_raw[features]
    df_val_raw = df_val_raw[features]
    df_test_raw = df_test_raw[features]
    df_diff_raw = df_diff_raw[features]

    # Preprocess data
    df_train_raw = preprocess_data(df_train_raw)
    df_val_raw = preprocess_data(df_val_raw)
    df_test_raw = preprocess_data(df_test_raw)
    df_diff_raw = preprocess_data(df_diff_raw)

    return df_train_raw, target_train, df_val_raw, target_val, df_test_raw, target_test, df_diff_raw, target_diff


def main():
    config = load_config()
    local_path = config['info']['local_path'] # type: ignore
    features = config['info']['numerical_features'] # type: ignore


    # Initialize logging
    logger = setup_logging(local_path, "LLM_Boost")
    original_stdout = sys.stdout
    logger_wrapper = LoggerWrapper(logger, original_stdout)
    sys.stdout = logger_wrapper

    logger.info("Starting LLM Boost Model Training")
    # Load data
    df_train_raw, target_train, df_val_raw, target_val, df_test_raw, target_test, df_diff_raw, target_diff = load_data(config)

    # Use training data for model fitting
    df_raw = df_train_raw
    target = target_train
    
    logger.info(f"Data loaded - Shape: {df_raw.shape}")
    logger.info(f"Features: {features}")
    logger.info(f"Target shape: {target.shape}")
    
    # Initialize models
    model_boost = LinearBoostModel(features, poly_degree=3)

    # Fit the initial model
    model_boost.fit_initial(df_raw, target)

    # Get the initial accuracy
    initial_accuracy = compute_metrics(model_boost.predict(df_raw), target, logger)[1] # type: ignore
    model_boost.set_current_accuracy(initial_accuracy)
    logger.info(f"Initial model accuracy: {initial_accuracy}")

    # On validation set accuracy
    val_accuracy = compute_metrics(model_boost.predict(df_val_raw), target_val, logger)[1] # type: ignore
    logger.info(f"Validation accuracy: {val_accuracy}")
    # On test set accuracy
    test_accuracy = compute_metrics(model_boost.predict(df_test_raw), target_test, logger)[1] # type: ignore
    logger.info(f"Test accuracy: {test_accuracy}")
    # On diff set accuracy
    diff_boost = compute_accuracy_in_bounds(model_boost.predict(df_diff_raw), target_diff)
    logger.info(f"Diff set accuracy: {diff_boost}")

    # Initialize tracking
    accuracy_boosting = [initial_accuracy]
    produced_features_dict = OrderedDict()
    queue_features_boosting = []  # Separate queue for boosting model
    new_features_this_iteration = []  # Track new features to evaluate
    start_iteration = 0

    best_boosting_accuracy = initial_accuracy
    best_boosting_model_state = capture_model_state(model_boost)

    # Checkpoint setup
    checkpoint_dir = os.path.join(local_path, 'boosting_checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'boosting_checkpoint.json')

    if os.path.exists(checkpoint_path):
        logger.info("Loading checkpoint...")
        checkpoint_data = load_checkpoint(checkpoint_path, model_boost)
        if checkpoint_data:
            start_iteration = checkpoint_data['iteration']
            accuracy_boosting = checkpoint_data['accuracy_boosting']
            produced_features_dict = OrderedDict(checkpoint_data['produced_features_dict'])
            # Load separate queues, with fallback for old checkpoints
            queue_features_boosting = checkpoint_data.get('queue_features_boosting', checkpoint_data.get('queue_features', []))
            # Load best model tracking from checkpoint
            best_boosting_accuracy = checkpoint_data.get('best_model_accuracy', max(accuracy_boosting) if accuracy_boosting else initial_accuracy)
            best_boosting_model_state = checkpoint_data.get('best_model_state', None)
            logger.info(f"Resumed from iteration {start_iteration}")
            logger.info(f"Best accuracies - Boost: {best_boosting_accuracy:.2f}%")
            logger.info(f"Current accuracies - Boost: {accuracy_boosting[-1]:.2f}%")

        else:
            logger.warning("Failed to load checkpoint, starting fresh")
            # Ensure best model states are captured for fresh start
            best_boosting_model_state = capture_model_state(model_boost)
    else:
        logger.info("No checkpoint found, starting fresh")
        # Ensure best model states are captured for fresh start
        best_boosting_model_state = capture_model_state(model_boost)

    # Load OpenAI client
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    
    logger.info("Starting feature generation and evaluation...")

    for i in range(start_iteration, 20):  # Start from checkpoint iteration
        logger.info(f"=== ITERATION {i+1} ===")
        
        new_features_this_iteration = []  # Reset for each iteration
        
        # Generate new feature
        prompt = build_feature_prompt(list(produced_features_dict.keys()), features)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = response.choices[0].message.content
            logger.info(f"Generated feature response: {content[:200]}...") # type: ignore
            
            code_str, reason, func, body = parse_feature_response(content, logger) # type: ignore
            logger.info(f"Parsed: code_str='{code_str}', reason='{reason}', body='{body}'")
            
            if code_str and code_str not in produced_features_dict:
                produced_features_dict[code_str] = {
                    'reason': reason, 
                    'func': func, 
                    'body': body,
                    'code_str': code_str
                }
                # Add to both queues so both models can evaluate it
                queue_features_boosting.append(code_str)
                new_features_this_iteration.append(code_str)  # Track new feature
                logger.info(f"✓ Added new feature to both queues: {code_str}")
            else:
                logger.warning(f"Duplicate or invalid feature skipped: {code_str}")
                logger.info(f"Existing features: {list(produced_features_dict.keys())}")
                continue
                
        except Exception as e:
            logger.error(f"Error generating feature: {e}")
            continue
        
        # Check if we have enough features before evaluating
        if len(new_features_this_iteration) == 0:
            logger.info('No new features generated this iteration, continuing...')
            continue
        
        # Add logic to ensure queue has at least 5 features before evaluation (like model_llm_boosting.py)
        if len(queue_features_boosting) < 5:
            logger.info(f'Not enough features in queues yet (Boosting: {len(queue_features_boosting)}), skipping evaluation.')
            continue
        
        # Feature selection logic with retries for both models
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            # Use batch evaluation for better performance
            iteration_best_boosting_accuracy = -1
            best_boosting_feature = None
            
            accuracy_list = []  # Store all evaluations for retry logic
            
            logger.info(f"Batch evaluating {len(new_features_this_iteration)} new features...")
            
            # Prepare feature list for batch evaluation
            feature_eval_list = []
            for code_str_eval in new_features_this_iteration:
                feature_info = produced_features_dict[code_str_eval]
                feature_name = generate_readable_feature_name(feature_info['body'])
                feature_eval_list.append((feature_name, feature_info['body']))
            
            # Batch evaluate for linear model
            logger.info("Evaluating features for Boost model...")
            Boost_results = model_boost.batch_evaluate_features(feature_eval_list, df_raw, target, logger)
            
            
            # Process boosting results
            for feature_name, feature_body, accuracy, residual_model, feature_predictor in Boost_results:
                logger.info(f"Boost Feature '{feature_name}': {accuracy:.2f}%")
                
                # Find corresponding code_str for this feature
                code_str_eval = None
                for code_str, info in produced_features_dict.items():
                    if info['body'] == feature_body:
                        code_str_eval = code_str
                        break
                
                if code_str_eval:
                    accuracy_list.append((feature_name, accuracy, code_str_eval, 
                                        produced_features_dict[code_str_eval]['func'], feature_body, 
                                        residual_model, feature_predictor))
                
                # Track best for boosting model
                if accuracy > iteration_best_boosting_accuracy:
                    iteration_best_boosting_accuracy = accuracy
                    best_boosting_feature = (feature_name, feature_body, residual_model, feature_predictor)


            # Feature selection logic - check if we found improvements
            boosting_improvement = iteration_best_boosting_accuracy >= model_boost.current_accuracy + 0.01

            if boosting_improvement:
                logger.info(f"Boosting model improvement found! {best_boosting_feature[0]}: {iteration_best_boosting_accuracy:.2f}%") # type: ignore
                model_boost.add_feature(best_boosting_feature[0], best_boosting_feature[1], best_boosting_feature[2], best_boosting_feature[3]) # type: ignore
                model_boost.set_current_accuracy(iteration_best_boosting_accuracy)
                accuracy_boosting.append(iteration_best_boosting_accuracy)
                    
                # Update global best if this is better
                if iteration_best_boosting_accuracy > best_boosting_accuracy:
                    best_boosting_accuracy = iteration_best_boosting_accuracy
                    best_boosting_model_state = capture_model_state(model_boost)
                    logger.info(f"New best boosting accuracy: {best_boosting_accuracy:.2f}%")
                
                # Remove selected features from their respective queues efficiently
                if best_boosting_feature:
                    selected_code = None
                    for item in accuracy_list:
                        if (item[1] == iteration_best_boosting_accuracy and 
                            item[4] == best_boosting_feature[1]):  # Match by feature body
                            selected_code = item[2]
                            break
                    if selected_code and selected_code in queue_features_boosting:
                        queue_features_boosting.remove(selected_code)
                        logger.info(f"Removed {selected_code} from boosting queue")
                
                break  # Exit retry loop
            else:
                logger.warning(f"No significant improvement found (Boosting: {iteration_best_boosting_accuracy:.2f}% vs {model_boost.current_accuracy:.2f}%)")

                if retry_count < max_retries - 1:
                    logger.info(f"Generating 3 additional features for retry {retry_count + 1}/{max_retries}...")
                    
                    # Generate additional features
                    additional_features_generated = 0
                    for attempt in range(6):
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.8 + retry_count * 0.1,
                            )
                            content = response.choices[0].message.content
                            logger.debug(f"Additional feature raw LLM response:\n{content}")
                            code_str, reason, func, body = parse_feature_response(content, logger) # type: ignore
                            
                            if code_str and code_str not in produced_features_dict:
                                produced_features_dict[code_str] = {
                                    'reason': reason, 
                                    'func': func, 
                                    'body': body,
                                    'code_str': code_str
                                }
                                # Add to both queues for evaluation
                                queue_features_boosting.append(code_str)
                                new_features_this_iteration.append(code_str)
                                additional_features_generated += 1
                                logger.debug(f"Generated additional feature {additional_features_generated}/3")
                                if additional_features_generated >= 3:
                                    break
                        except Exception as e:
                            logger.error(f"Error generating additional feature: {e}")
                            continue
                    
                    if additional_features_generated == 0:
                        logger.warning("Could not generate any additional features, stopping retries.")
                        break
                else:
                    logger.warning(f"No improvement found after {max_retries} retries. Moving to next iteration.")
                    if not boosting_improvement:
                        accuracy_boosting.append(model_boost.current_accuracy)
                    break
                    
            retry_count += 1
            
            # Efficiently evaluate on validation/test sets only if improvements were made
            if boosting_improvement:
                # Evaluate on validation set    
                if boosting_improvement:
                    val_boosting = compute_metrics(model_boost.predict(df_val_raw), target_val, logger)[1] # type: ignore
                    test_boosting = compute_metrics(model_boost.predict(df_test_raw), target_test, logger)[1] # type: ignore
                    diff_boosting = compute_accuracy_in_bounds(model_boost.predict(df_diff_raw), target_diff) # type: ignore
                    logger.info(f"Validation Boosting accuracy: {val_boosting:.2f}%")
                    logger.info(f"Test Boosting accuracy: {test_boosting:.2f}%")
                    logger.info(f"Diff Boosting accuracy: {diff_boosting:.2f}%")
        
        # Limit queue sizes to prevent memory issues and reduce redundant computation
        max_queue_size = 15  # Reduced from 20 for better performance
            
        if len(queue_features_boosting) > max_queue_size:
            # Keep most recent features and some random older ones for diversity
            recent_features = queue_features_boosting[-max_queue_size//2:]
            older_features = queue_features_boosting[:-max_queue_size//2]
            if older_features:
                selected_older = np.random.choice(older_features, 
                                                size=min(max_queue_size//2, len(older_features)), 
                                                replace=False).tolist()
                queue_features_boosting = selected_older + recent_features
            else:
                queue_features_boosting = recent_features
            logger.info(f"Optimized boosting queue to {len(queue_features_boosting)} features")
        
        # Save checkpoint after each iteration
        checkpoint_state = {
            'iteration': i + 1,
            'accuracy_boosting': accuracy_boosting,
            'produced_features_dict': produced_features_dict,
            'queue_features_boosting': queue_features_boosting,
            'model_boosting': model_boost,
            'best_boosting_accuracy': best_boosting_accuracy,
            'best_boosting_model_state': best_boosting_model_state
        }
        save_checkpoint(checkpoint_state, checkpoint_path)

    # Final results and cleanup
    logger.info("="*60)
    logger.info("FINAL RESULTS AND COMPARISON")
    logger.info("="*60)
    
    logger.info(f"Total features produced: {len(produced_features_dict)}")
    logger.info(f"Boosting model features: {len(model_boost.selected_features)}")
    logger.info(f"Boosting queue remaining: {len(queue_features_boosting)} features")

    # Final model accuracies
    final_boosting_accuracy = accuracy_boosting[-1] if accuracy_boosting else model_boost.current_accuracy
    max_boosting_accuracy = best_boosting_accuracy  # Use tracked best instead of max(accuracy_boosting)

    logger.info(f"Final Boosting model accuracy: {final_boosting_accuracy:.2f}%")
    logger.info(f"Maximum Boosting accuracy achieved: {max_boosting_accuracy:.2f}%")
    logger.info(f"Boosting improvement over baseline: {max_boosting_accuracy - initial_accuracy:.2f}%")

    # Accuracy on validation set
    val_boosting_final = compute_metrics(model_boost.predict(df_val_raw), target_val, logger)[1] # type: ignore
    logger.info(f"Validation Boosting final accuracy: {val_boosting_final:.2f}%")
    # Accuracy on test set
    test_boosting_final = compute_metrics(model_boost.predict(df_test_raw), target_test, logger)[1] # type: ignore
    logger.info(f"Test Boosting final accuracy: {test_boosting_final:.2f}%")
    # Accuracy on diff set
    diff_boosting_final = compute_accuracy_in_bounds(model_boost.predict(df_diff_raw), target_diff)
    logger.info(f"Diff Boosting final accuracy: {diff_boosting_final:.2f}%")

    plot_graph(accuracy_boosting, "model_accuracy.png", local_path=local_path)

    # Save all results including models, features, and evaluation metrics
    save_results(model_boost, accuracy_boosting, 
                produced_features_dict, "final_results", logger,
                df_train_raw, df_val_raw, df_test_raw, 
                target_train, target_val, target_test, local_path)

    # Save accuracy progress to CSV
    max_len = len(accuracy_boosting)
    final_df = pd.DataFrame({
        'iteration': range(1, max_len + 1),
        'boosting_accuracy': accuracy_boosting
    })
    final_df.to_csv(os.path.join(local_path, "model_progress.csv"), index=False)

    logger.info("Model comparison optimization complete!")
    
    # Cleanup logging
    logger.info("="*60)
    logger.info("MODEL COMPARISON EXECUTION COMPLETED")
    logger.info("="*60)

    # Restore original stdout
    sys.stdout = original_stdout

    # Log the final location of the log file
    log_file_path = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file_path = handler.baseFilename
            break

    if log_file_path:
        print(f"Full execution log saved to: {log_file_path}")
    else:
        print("Log file location not found")


if __name__ == "__main__":
    main()