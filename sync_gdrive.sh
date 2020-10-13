#!/usr/bin/env bash
rclone sync music_decomposition_gdrive_remote:/notebooks notebooks
rclone sync libs music_decomposition_gdrive_remote:/libs
