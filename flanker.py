import pickle
import numpy as np
import mne
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

def load_flanker_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        flanker_data = pickle.load(file)
    return flanker_data

class Participant:
    def __init__(self, participant_id=None, eeg_data=None, labels=None):
        self.raw_unprocessed = None
        self.raw_processed = None
        self.id = participant_id
        self.X = eeg_data
        self.Y = labels
        self.epochs = None
        self.channels = None
        self.psd = None

class FlankerData:
    def __init__(self):
        self.participants = {}
        self.num_participants = 0
        self.epochs = None

    def add_participant(self, participant_id, eeg_data, labels):
        participant = Participant(participant_id, eeg_data, labels)
        self.participants[participant_id] = participant
        self.num_participants += 1

    def add_participant_path(self, filepath, id):
        if id in self.participants.keys():
            print("Duplicate participant ID: already taken.")
        else:
            participant = self._process_participant(filepath, id)
            self.participants[id] = participant
            print(f"Added participant {participant.id}")

    def get_participant(self, participant_id):
        return self.participants.get(participant_id, None)

    def save_to_pickle(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def split_data(self):
        # Extract participant IDs and corresponding EEG data

        all_eeg_data = []
        all_labels = []
        participant_ids_per_sample = []
        participant_id = 0
        counter = 0
        for participant_data in self.participants.values():
            eeg_data = participant_data.X
            labels = participant_data.Y

            all_eeg_data.append(eeg_data)
            all_labels.extend(labels)
            # print(participant_data.channels)

            # Assign participant ID to each sample
            participant_ids_per_sample.extend([participant_id] * len(eeg_data))
            counter += 1
            if counter % 3 == 0:
                participant_id += 1

        # Find the minimum number of time points across all epochs
        min_time_points = min(epoch.shape[2] for epoch in all_eeg_data)

        # Extract the first min_time_points from each epoch
        all_eeg_data_trimmed = [epoch[:, :, :min_time_points] for epoch in all_eeg_data]

        X = np.concatenate(all_eeg_data_trimmed, axis=0)
        y = np.array(all_labels)

        X_reshaped = X.reshape((X.shape[0], -1))

        # Use GroupShuffleSplit to split the data without participant leakage
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        # Perform the split
        train_indices, test_indices = next(gss.split(X_reshaped, y, groups=participant_ids_per_sample))

        # Create training and test sets
        X_train, Y_train = X_reshaped[train_indices], y[train_indices]
        X_test, Y_test = X_reshaped[test_indices], y[test_indices]

        return X_train, Y_train, X_test, Y_test

    def concatenate_data(self):
        all_eeg_data = []
        all_labels = []

        for participant_data in self.participants.values():
            eeg_data = participant_data.X
            labels = participant_data.Y

            all_eeg_data.append(eeg_data)
            all_labels.extend(labels)

        X = np.concatenate(all_eeg_data, axis=0)
        y = np.array(all_labels)

        return X, y

    def process_eeg_data(self, raw):
        # Define the filter settings
        low_freq = 1.0  # 1Hz
        high_freq = None  # None for high-pass filter
        filter_length = 'auto'  # Automatically choose the filter length
        fir_design = 'firwin'  # FIR filter design method

        # Apply high-pass FIR filter
        raw.filter(l_freq=low_freq, h_freq=high_freq, filter_length=filter_length, fir_design=fir_design)

        # Define the line noise frequency to filter out (50 Hz for European systems)
        line_freq = 50.0

        # Apply a notch filter to remove the specified line noise frequency
        raw.notch_filter(freqs=[line_freq], fir_design='firwin')

        # RESAMPLE TO 250Hz
        # Define the target sampling frequency (e.g., 250 Hz)
        target_sf = 250

        # Downsample the raw data
        raw.resample(sfreq=target_sf)

        return raw

        # # REJECT CHANNELS
        # # Automatic channel rejection and interpolation are performed based on 2 standard deviations for EEG channels.
        # # You can adjust the reject dictionary to set rejection thresholds for other channel types if needed.
        #
        # # Perform automatic rejection based on 2 standard deviations
        # reject = dict(eeg=2e-4)  # Set the threshold for EEG channels (adjust as needed)
        #
        # # Automatically detect bad channels and reject epochs
        # reject, _ = mne.preprocessing.compute_reject_data(raw, reject=reject)
        #
        # # Drop or interpolate bad channels
        # if reject['eeg']:
        #     print(f"Detected bad EEG channels for Participant {self.participant_id}: {reject['eeg']}")
        #     # Drop bad channels
        #     raw.drop_channels(reject['eeg'])
        #     # Alternatively, you can interpolate bad channels instead of dropping them
        #     # raw.interpolate_bads()
        # else:
        #     print("No bad EEG channels detected.")

    def _get_events(self, raw, inverted_event_ids, event_id, option):

        # Extract events from raw data
        events, _ = mne.events_from_annotations(raw, inverted_event_ids)

        if option == 1:

            # Define epochs starting right after events '241' and '242'
            tmin, tmax = 0.0, 0.5  # define the time window for each epoch

            # Find the indices of events '241' and '242'
            event_241_indices = np.where(events[:, -1] == event_id['FLANKER_Stimulus_cong'])[0]
            event_242_indices = np.where(events[:, -1] == event_id['FLANKER_Stimulus_incong'])[0]

            # Concatenate the indices of events '241' and '242'
            start_indices = np.concatenate([event_241_indices, event_242_indices])

            # Create epochs for each start index
            epochs_list = []
            labels = []

            for start_index in start_indices:
                # Determine the label based on the event following '241' or '242'
                next_event = events[start_index + 1, -1]
                if next_event == event_id['FLANKER_Response_Correct_cong']:
                    label = 1
                elif next_event == event_id['FLANKER_Response_Incorrect_cong']:
                    label = 0
                elif next_event == event_id['FLANKER_Response_Correct_incong']:
                    label = 1
                elif next_event == event_id['FLANKER_Response_Incorrect_incong']:
                    label = 0
                else:
                    label = 'Unknown'

                # Create an epoch for the specified time window
                epoch = mne.Epochs(raw, events=np.array([events[start_index]]),
                                   event_id=None,
                                   tmin=tmin,
                                   tmax=tmax,
                                   baseline=None,
                                   preload=True)


                # Append the epoch to the list
                epochs_list.append(epoch)
                labels.append(label)

            # Concatenate the list of epochs into a single Epochs object
            epochs = mne.concatenate_epochs(epochs_list)

            return epochs, labels

        if option == 2:
            # Define epochs starting right after events '241' and '242'
            tmin, tmax = -0.5, 0.0  # define the time window for each epoch

            # Find the indices of events '241' and '242'
            event_correct_cong = np.where(events[:, -1] == event_id['FLANKER_Response_Correct_cong'])[0]
            event_incorrect_cong = np.where(events[:, -1] == event_id['FLANKER_Response_Incorrect_cong'])[0]
            event_correct_incong = np.where(events[:, -1] == event_id['FLANKER_Response_Correct_incong'])[0]
            event_incorrect_incong = np.where(events[:, -1] == event_id['FLANKER_Response_Incorrect_incong'])[0]


            # Concatenate the indices of all events
            start_indices = np.concatenate([event_correct_cong,
                                            event_incorrect_cong,
                                            event_correct_incong,
                                            event_incorrect_incong])


            # Create epochs for each start index
            epochs_list = []
            labels = []

            for start_index in start_indices:
                current_event = events[start_index, -1]
                if current_event == event_id['FLANKER_Response_Correct_cong']:
                    label = 1
                elif current_event == event_id['FLANKER_Response_Incorrect_cong']:
                    label = 0
                elif current_event == event_id['FLANKER_Response_Correct_incong']:
                    label = 1
                elif current_event == event_id['FLANKER_Response_Incorrect_incong']:
                    label = 0
                else:
                    label = 'Unknown'

                # Create an epoch for the specified time window
                epoch = mne.Epochs(raw, events=np.array([events[start_index]]),
                                   event_id=None,
                                   tmin=tmin,
                                   tmax=tmax,
                                   baseline=None,
                                   preload=True)


                # Append the epoch to the list
                epochs_list.append(epoch)
                labels.append(label)

            # Concatenate the list of epochs into a single Epochs object
            epochs = mne.concatenate_epochs(epochs_list)
            psd = epochs.compute_psd(method='multitaper', fmin=0, fmax=50)

            return epochs, labels, psd

    def _process_participant(self, filepath, id):

        participant = Participant()
        participant.id = id

        # Load the .set EEG file
        # eeg_file = Path("C:\\Users\\massimo\Desktop\\7413650\\Dataset\\COG-BCI\\sub-01\\ses-S1\\eeg\\Flanker.set")
        # fdt_file = Path("C:\\Users\\massimo\Desktop\\7413650\\Dataset\\COG-BCI\\sub-01\\ses-S1\\eeg\\Flanker.fdt")
        # eeg_file = 'path/to/your/file.set'
        participant.raw = mne.io.read_raw_eeglab(str(filepath), preload=True) # FDT file also loaded if same directory

        participant.channels = participant.raw.ch_names

        processed_raw = self.process_eeg_data(participant.raw)
        participant.raw_processed = processed_raw

        # Specify the event IDs and descriptions
        event_id = {
            'FLANKER_Start': 20,
            'FLANKER_Trial/ISI_Start': 210,
            'FLANKER_Error_ISI': 221,
            'FLANKER_Error_FIXI': 222,
            'FLANKER_Fixation_Cross': 23,
            'FLANKER_Stimulus_cong': 241,
            'FLANKER_Response_Correct_cong': 2511,
            'FLANKER_Response_Incorrect_cong': 2521,
            'FLANKER_Response_Correct_Feedback_cong': 25121,
            'FLANKER_Response_Incorrect_Feedback_cong': 25221,
            'FLANKER_Missed_Response_Feedback_cong': 25321,
            'FLANKER_End': 21,
            'FLANKER_Stimulus_incong': 242,
            'FLANKER_Response_Correct_incong': 2512,
            'FLANKER_Response_Incorrect_incong': 2522,
            'FLANKER_Response_Correct_Feedback_incong': 25122,
            'FLANKER_Response_Incorrect_Feedback_incong': 25222,
            'FLANKER_Missed_Response_Feedback_incong': 25322,
        }
        flanker_id_names = {
            20: 'FLANKER_Start',
            210: 'FLANKER_Trial/ISI_Start',
            221: 'FLANKER_Error_ISI',
            222: 'FLANKER_Error_FIXI',
            23: 'FLANKER_Fixation_Cross',
            241: 'FLANKER_Stimulus_cong',
            2511: 'FLANKER_Response_Correct_cong',
            2521: 'FLANKER_Response_Incorrect_cong',
            25121: 'FLANKER_Response_Correct_Feedback_cong',
            25221: 'FLANKER_Response_Incorrect_Feedback_cong',
            25321: 'FLANKER_Missed_Response_Feedback_cong',
            21: 'FLANKER_End',
            242: 'FLANKER_Stimulus_incong',
            2512: 'FLANKER_Response_Correct_incong',
            2522: 'FLANKER_Response_Incorrect_incong',
            25122: 'FLANKER_Response_Correct_Feedback_incong',
            25222: 'FLANKER_Response_Incorrect_Feedback_incong',
            25322: 'FLANKER_Missed_Response_Feedback_incong',
        }

        inverted_event_id = {
            '20': 20,
            '210': 210,
            '221': 221,
            '222': 222,
            '23': 23,
            '241': 241,
            '2511': 2511,
            '2521': 2521,
            '25121': 25121,
            '25221': 25221,
            '25321': 25321,
            '21': 21,
            '242': 242,
            '2512': 2512,
            '2522': 2522,
            '25122': 25122,
            '25222': 25222,
            '25322': 25322,
        }

        # Option 1 = 0.5 seconds after stimulus
        # Option 2 = 0.5 seconds before action
        epochs, labels, psd = self._get_events(processed_raw, inverted_event_id, event_id, 2)

        # List of channels to keep
        ignored_channels = ['Cz', 'ECG1']
        channels_to_keep = []
        for ch in epochs.ch_names:
            if not ch in ignored_channels:
                channels_to_keep.append(ch)

        # Pick the desired channels
        epochs_cleaned = epochs.pick_channels(channels_to_keep)
        psd_cleaned = psd.pick_channels(channels_to_keep)

        print(f"EEG Channels: {epochs.ch_names}")
        print(f"PSD Channels: {psd.ch_names}")
        eeg_data_extracted = epochs_cleaned.get_data()
        psd_data_extracted = psd_cleaned.get_data()

        # Extract data and labels for machine learning
        participant.X = np.concatenate([eeg_data_extracted, psd_data_extracted], axis=2)
        participant.Y = labels  # Event labels
        participant.psd = psd
        participant.epochs = epochs_cleaned
        print(f"Done processing participant {participant.id}")
        return participant

