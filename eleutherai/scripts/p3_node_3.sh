for TASK in ag_news_classify amazon_polarity_Is_this_review_negative amazon_polarity_negative_or_positive_tone app_reviews_convert_to_rating cnn_dailymail_3.0.0_news_card_view cnn_dailymail_3.0.0_sum_in_brief common_gen_Given_concepts_type_1 common_gen_random_task_template_prompt cos_e_v1.11_aligned_with_common_sense cos_e_v1.11_generate_explanation_given_text cos_e_v1.11_question_option_description_id cosmos_qa_context_description_question_answer_id cosmos_qa_context_question_description_answer_text cosmos_qa_description_context_question_text dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to dream_baseline duorc_ParaphraseRC_answer_question duorc_ParaphraseRC_generate_question duorc_ParaphraseRC_title_generation duorc_SelfRC_extract_answer duorc_SelfRC_question_answering gigaword_generate_summary_for_this gigaword_write_a_title_for_this_sentence glue_mrpc_generate_paraphrase glue_mrpc_same_thing glue_qqp_duplicate_or_not imdb_Movie_Expressed_Sentiment imdb_Reviewer_Enjoyment_Yes_No imdb_Sentiment_with_choices_ multi_news_expand_reverse_task_ multi_news_what_are_the_key_points paws_labeled_final_Meaning_no_label paws_labeled_final_Rewrite_no_label paws_labeled_final_task_description_no_label qasc_qa_with_separated_facts_1 qasc_qa_with_separated_facts_5 quail_context_question_answer_description_id quail_context_question_description_text quail_no_prompt_id quarel_heres_a_story quartz_answer_question_below quartz_read_passage_below_choose quoref_Answer_Question_Given_Context quoref_Found_Context_Online quoref_Read_And_Extract_ ropes_given_background_situation ropes_plain_no_background ropes_prompt_mix rotten_tomatoes_Reviewer_Enjoyment rotten_tomatoes_Reviewer_Sentiment_Feeling samsum_Generate_a_summary_for_this_dialogue samsum_Summarize_this_dialogue_ sciq_Direct_Question_Closed_Book_ social_i_qa_Check_if_a_random_answer_is_valid_or_not social_i_qa_Show_choices_and_generate_answer trec_fine_grained_DESC trec_fine_grained_HUM_context_first trec_fine_grained_NUM_context_first trec_trec1 wiki_bio_comprehension wiki_bio_who wiki_hop_original_choose_best_object_interrogative_1 wiki_hop_original_generate_subject wiki_qa_Generate_Question_from_Topic wiki_qa_Topic_Prediction_Question_Only wiki_qa_found_on_google wiqa_what_is_the_final_step_of_the_following_process wiqa_which_of_the_following_is_the_supposed_perturbation xsum_DOC_tldr xsum_read_below_DOC_write_abstract yelp_review_full_format_rating yelp_review_full_so_i_would; do
    PYTHONPATH=/home/connor/code/FLAN/:${PYTHONPATH} \
        python3 -m seqio.scripts.cache_tasks_main \
        --tasks=${TASK} \
        --output_cache_dir=gs://neo-datasets/zphang/multitask/t0_seqio/v1 \
        --module_import=promptsource.seqio_tasks \
        --alsologtostderr
done