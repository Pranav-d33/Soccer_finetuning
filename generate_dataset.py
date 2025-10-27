"""
Soccer Dataset Preprocessing & Instruction Generation
Converts HugoMathien European Soccer Database into a high-quality instruction dataset
for fine-tuning language models on tactical reasoning and match analysis.
"""

import sqlite3
import pandas as pd
import json
import random
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class SoccerDatasetGenerator:
    def __init__(self, db_path: str = "data/database.sqlite", output_dir: str = "."):
        """Initialize the dataset generator."""
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.conn = None
        
    def connect_db(self):
        """Connect to SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def load_data(self):
        """Load all necessary tables from the database."""
        logger.info("Loading data tables...")
        
        try:
            # Load main tables
            self.matches = pd.read_sql_query("SELECT * FROM Match", self.conn)
            self.teams = pd.read_sql_query("SELECT * FROM Team", self.conn)
            self.team_attr = pd.read_sql_query("SELECT * FROM Team_Attributes", self.conn)
            self.players = pd.read_sql_query("SELECT * FROM Player", self.conn)
            self.leagues = pd.read_sql_query("SELECT * FROM League", self.conn)
            
            logger.info(f"Matches loaded: {len(self.matches)} records")
            logger.info(f"Teams loaded: {len(self.teams)} records")
            logger.info(f"Team Attributes loaded: {len(self.team_attr)} records")
            logger.info(f"Players loaded: {len(self.players)} records")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_matches(self):
        """Preprocess match data with team attributes."""
        logger.info("Preprocessing match data...")
        
        # Select relevant columns from matches
        df = self.matches[[
            "id", "season", "home_team_api_id", "away_team_api_id",
            "home_team_goal", "away_team_goal", "country_id",
            "date"
        ]].copy()
        
        # Select relevant team attributes, excluding date to avoid duplication
        team_attr_cols = [
            "team_api_id",
            "buildUpPlaySpeed", "buildUpPlayPassing",
            "buildUpPlayDribbling",
            "chanceCreationPassing", "chanceCreationShooting", "chanceCreationCrossing",
            "defencePressure", "defenceAggression", "defenceTeamWidth"
        ]
        
        # Only keep columns that exist
        team_attr_cols = [col for col in team_attr_cols if col in self.team_attr.columns]
        
        # For home team attributes
        home_team_attr = self.team_attr[team_attr_cols].copy()
        home_team_attr = home_team_attr.rename(columns={col: f"{col}_home" for col in team_attr_cols if col != "team_api_id"})
        
        # For away team attributes
        away_team_attr = self.team_attr[team_attr_cols].copy()
        away_team_attr = away_team_attr.rename(columns={col: f"{col}_away" for col in team_attr_cols if col != "team_api_id"})
        
        # Merge home team attributes
        df = df.merge(
            home_team_attr,
            left_on="home_team_api_id",
            right_on="team_api_id",
            how="left"
        )
        df = df.drop(columns=["team_api_id"])
        
        # Merge away team attributes
        df = df.merge(
            away_team_attr,
            left_on="away_team_api_id",
            right_on="team_api_id",
            how="left"
        )
        df = df.drop(columns=["team_api_id"])
        
        # Merge team names for home team
        home_teams = self.teams[['team_api_id', 'team_long_name']].copy()
        home_teams = home_teams.rename(columns={'team_long_name': 'home_team_name', 'team_api_id': 'home_team_id'})
        df = df.merge(
            home_teams,
            left_on='home_team_api_id',
            right_on='home_team_id',
            how='left'
        )
        df = df.drop(columns=['home_team_id'])
        
        # Merge team names for away team
        away_teams = self.teams[['team_api_id', 'team_long_name']].copy()
        away_teams = away_teams.rename(columns={'team_long_name': 'away_team_name', 'team_api_id': 'away_team_id'})
        df = df.merge(
            away_teams,
            left_on='away_team_api_id',
            right_on='away_team_id',
            how='left'
        )
        df = df.drop(columns=['away_team_id'])
        
        # Merge league names
        leagues_merged = self.leagues[['id', 'name']].copy()
        leagues_merged = leagues_merged.rename(columns={'id': 'league_id', 'name': 'league_name'})
        df = df.merge(
            leagues_merged,
            left_on='country_id',
            right_on='league_id',
            how='left'
        )
        df = df.drop(columns=['league_id'], errors='ignore')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['home_team_name', 'away_team_name', 'buildUpPlaySpeed_home', 'buildUpPlaySpeed_away'])
        
        self.processed_matches = df
        logger.info(f"Matches after preprocessing: {len(df)} records")
        return df
    
    def describe_playstyle(self, speed: float, passing: float = None, aggression: float = None) -> str:
        """Convert numeric attributes to qualitative descriptors."""
        if pd.isna(speed):
            return "unknown"
        
        speed = float(speed)
        
        if speed < 45:
            base_style = "slow, methodical buildup"
        elif speed < 70:
            base_style = "balanced, controlled pace"
        else:
            base_style = "fast, direct counterattack"
        
        return base_style
    
    def describe_tactical_approach(self, passing: float, aggression: float, width: float) -> str:
        """Describe overall tactical approach based on attributes."""
        descriptions = []
        
        if not pd.isna(passing):
            if passing > 60:
                descriptions.append("short-pass oriented")
            else:
                descriptions.append("direct play")
        
        if not pd.isna(aggression):
            if aggression > 60:
                descriptions.append("aggressive pressing")
            else:
                descriptions.append("defensive discipline")
        
        if not pd.isna(width):
            if width > 60:
                descriptions.append("wide attacking play")
            else:
                descriptions.append("central focus")
        
        return ", ".join(descriptions) if descriptions else "conservative approach"
    
    def generate_tactical_reasoning(self, row: pd.Series) -> dict:
        """Generate tactical reasoning instruction."""
        home_team = row['home_team_name']
        away_team = row['away_team_name']
        score = f"{row['home_team_goal']}-{row['away_team_goal']}"
        season = row.get('season', 'unknown season')
        league = row.get('league_name', 'League')
        
        home_style = self.describe_playstyle(row.get('buildUpPlaySpeed_home'))
        away_style = self.describe_playstyle(row.get('buildUpPlaySpeed_away'))
        
        home_tactics = self.describe_tactical_approach(
            row.get('buildUpPlayPassing_home'),
            row.get('defenceAggression_home'),
            row.get('defenceTeamWidth_home')
        )
        
        instructions = [
            f"Explain why {home_team} used their {home_style} approach against {away_team}.",
            f"Analyze the tactical setup of {home_team} in their {score} match.",
            f"Describe {home_team}'s tactical adjustments for the {score} result."
        ]
        
        outputs = [
            f"{home_team} employed a {home_style} strategy to dominate possession and control the game tempo. "
            f"Their {home_tactics} approach was designed to neutralize {away_team}'s strengths and build attacks methodically. "
            f"With the final score of {score}, their tactical execution proved effective in key moments.",
            
            f"{home_team} set up with a {home_style} formation and {home_tactics}. "
            f"This tactical choice reflected their intention to control the midfield and create sustained pressure. "
            f"Against {away_team}'s {away_style} defense, they achieved a {score} scoreline through disciplined execution.",
            
            f"In their {score} result, {home_team} demonstrated {home_style} buildup play combined with {home_tactics}. "
            f"The tactical adjustment showed adaptability to {away_team}'s defensive shape. "
            f"Their ability to maintain shape and transition quickly was crucial to the outcome."
        ]
        
        instruction = random.choice(instructions)
        output = random.choice(outputs)
        
        return {
            "instruction": instruction,
            "input": f"Match: {home_team} vs {away_team} | Season: {season} | League: {league} | Score: {score}",
            "output": output,
            "metadata": {
                "match_id": row['id'],
                "home_team": home_team,
                "away_team": away_team,
                "type": "tactical_reasoning"
            }
        }
    
    def generate_match_summary(self, row: pd.Series) -> dict:
        """Generate match summary instruction."""
        home_team = row['home_team_name']
        away_team = row['away_team_name']
        home_goals = int(row['home_team_goal'])
        away_goals = int(row['away_team_goal'])
        score = f"{home_goals}-{away_goals}"
        
        home_style = self.describe_playstyle(row.get('buildUpPlaySpeed_home'))
        away_style = self.describe_playstyle(row.get('buildUpPlaySpeed_away'))
        
        if home_goals > away_goals:
            result = f"{home_team} won"
            momentum = "shifted in their favor"
            winner = home_team
        elif home_goals < away_goals:
            result = f"{away_team} won"
            momentum = "favored the away side"
            winner = away_team
        else:
            result = "The match ended in a draw"
            momentum = "remained balanced"
            winner = "Neither team"
        
        instruction = f"Summarize the match between {home_team} and {away_team}."
        
        output = (
            f"{result} with a final score of {score}. {home_team} approached the match with {home_style} play, "
            f"while {away_team} countered with {away_style} tactics. The momentum {momentum} as both teams "
            f"battled for control throughout the 90 minutes. {winner} capitalized on key opportunities to secure the result."
        )
        
        return {
            "instruction": instruction,
            "input": f"{home_team} vs {away_team}",
            "output": output,
            "metadata": {
                "match_id": row['id'],
                "home_team": home_team,
                "away_team": away_team,
                "type": "match_summary"
            }
        }
    
    def generate_formation_comparison(self, row: pd.Series) -> dict:
        """Generate formation comparison instruction."""
        home_team = row['home_team_name']
        away_team = row['away_team_name']
        score = f"{row['home_team_goal']}-{row['away_team_goal']}"
        
        home_tactics = self.describe_tactical_approach(
            row.get('buildUpPlayPassing_home'),
            row.get('defenceAggression_home'),
            row.get('defenceTeamWidth_home')
        )
        
        away_tactics = self.describe_tactical_approach(
            row.get('buildUpPlayPassing_away'),
            row.get('defenceAggression_away'),
            row.get('defenceTeamWidth_away')
        )
        
        instruction = f"Compare the tactical formations of {home_team} and {away_team} in this match."
        
        output = (
            f"{home_team} deployed a {home_tactics} formation to control play. "
            f"In contrast, {away_team} opted for a {away_tactics} setup to counter-attack and defend. "
            f"The tactical contrast resulted in a {score} scoreline, highlighting how formation choices impacted the match outcome. "
            f"The way each team balanced offense and defense was crucial in determining possession and scoring opportunities."
        )
        
        return {
            "instruction": instruction,
            "input": f"{home_team} ({home_tactics}) vs {away_team} ({away_tactics})",
            "output": output,
            "metadata": {
                "match_id": row['id'],
                "home_team": home_team,
                "away_team": away_team,
                "type": "formation_comparison"
            }
        }
    
    def generate_predictive_analysis(self, row: pd.Series) -> dict:
        """Generate what-if/predictive analysis instruction."""
        home_team = row['home_team_name']
        away_team = row['away_team_name']
        home_goals = int(row['home_team_goal'])
        away_goals = int(row['away_team_goal'])
        
        # Only generate if there's a potential "what-if" scenario
        if home_goals > away_goals:
            scenario_team = away_team
            scenario_question = f"What tactical adjustments could {away_team} have made to change the {home_goals}-{away_goals} outcome?"
            
            output = (
                f"To overcome the {home_goals}-{away_goals} deficit, {away_team} could have: "
                f"(1) increased pressing intensity to disrupt {home_team}'s buildup, "
                f"(2) shifted to a more aggressive defensive formation with additional attacking players, "
                f"(3) exploited set-piece opportunities more effectively, and "
                f"(4) adjusted their width to create more space for counter-attacks. "
                f"These tactical modifications might have created more scoring chances and altered the match result."
            )
        elif away_goals > home_goals:
            scenario_team = home_team
            scenario_question = f"What tactical adjustments could {home_team} have made to change the {home_goals}-{away_goals} outcome?"
            
            output = (
                f"To overcome the {home_goals}-{away_goals} deficit, {home_team} could have: "
                f"(1) accelerated their buildup play to create faster transitions, "
                f"(2) increased the tempo and aggression in midfield battles, "
                f"(3) deployed additional attacking support in wide areas to create crossing opportunities, and "
                f"(4) adjusted their defensive shape to invite controlled attacks. "
                f"These tactical shifts might have provided better balance and more goal-scoring opportunities."
            )
        else:
            # For draws, ask both teams
            scenario_team = home_team
            scenario_question = f"What tactical adjustments could either team have made to win the {home_goals}-{away_goals} draw?"
            
            output = (
                f"To break the deadlock in a {home_goals}-{away_goals} draw, either team could have: "
                f"(1) increased pressing intensity to force turnovers in dangerous areas, "
                f"(2) adjusted formation to create numerical advantages in key zones, "
                f"(3) introduced fresh tactical ideas through substitutions, and "
                f"(4) utilized set-pieces more strategically. "
                f"Small tactical adjustments in timing and spacing often determine the outcome in tight matches."
            )
        
        return {
            "instruction": scenario_question,
            "input": f"{home_team} vs {away_team} | Score: {home_goals}-{away_goals}",
            "output": output,
            "metadata": {
                "match_id": row['id'],
                "home_team": home_team,
                "away_team": away_team,
                "type": "predictive_analysis"
            }
        }
    
    def generate_player_analysis(self, row: pd.Series) -> dict:
        """Generate player performance analysis instruction."""
        home_team = row['home_team_name']
        away_team = row['away_team_name']
        home_goals = int(row['home_team_goal'])
        
        # Create generic player analysis (since we don't have detailed player stats per match)
        if home_goals > 0:
            analysis_team = home_team
            context = f"scored {home_goals} goals"
        else:
            analysis_team = away_team
            context = "demonstrated strong defensive organization"
        
        instruction = f"Analyze the key performance factors that led to {analysis_team}'s result in this match."
        
        output = (
            f"{analysis_team} showed tactical discipline throughout the match. "
            f"Their players executed the formation well, maintaining shape during transitions. "
            f"The midfield controlled tempo effectively, and defensive line maintained compactness to prevent scoring chances. "
            f"Attacking players made intelligent runs and positioning decisions. "
            f"Overall, the team's collective effort in following tactical instructions was evident in their performance."
        )
        
        return {
            "instruction": instruction,
            "input": f"{home_team} vs {away_team}",
            "output": output,
            "metadata": {
                "match_id": row['id'],
                "home_team": home_team,
                "away_team": away_team,
                "type": "player_analysis"
            }
        }
    
    def generate_instructions(self, max_samples_per_match: int = 4):
        """Generate all instruction types for each match."""
        logger.info(f"Generating instructions (max {max_samples_per_match} per match)...")
        
        samples = []
        instruction_types = [
            self.generate_tactical_reasoning,
            self.generate_match_summary,
            self.generate_formation_comparison,
            self.generate_predictive_analysis,
            self.generate_player_analysis
        ]
        
        for idx, (_, row) in enumerate(self.processed_matches.iterrows()):
            # Randomly select instruction types for diversity
            selected_types = random.sample(instruction_types, min(max_samples_per_match, len(instruction_types)))
            
            for instr_func in selected_types:
                try:
                    sample = instr_func(row)
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Error generating instruction for match {row['id']}: {e}")
                    continue
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1} matches, generated {len(samples)} samples")
        
        logger.info(f"Total samples generated: {len(samples)}")
        return samples
    
    def split_and_save(self, samples: list, train_ratio: float = 0.9):
        """Split data into train/validation and save as JSONL."""
        logger.info(f"Splitting data: {train_ratio*100}% train, {(1-train_ratio)*100}% validation")
        
        random.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)
        
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # Save training set
        train_file = self.output_dir / "football_train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Training set saved: {train_file} ({len(train_samples)} samples)")
        
        # Save validation set
        val_file = self.output_dir / "football_val.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Validation set saved: {val_file} ({len(val_samples)} samples)")
        
        return train_file, val_file
    
    def generate_statistics(self, samples: list):
        """Generate and log dataset statistics."""
        logger.info("\n=== Dataset Statistics ===")
        
        total_samples = len(samples)
        logger.info(f"Total samples: {total_samples}")
        
        # Count by type
        type_counts = {}
        for sample in samples:
            sample_type = sample.get('metadata', {}).get('type', 'unknown')
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
        
        logger.info("Samples by type:")
        for sample_type, count in sorted(type_counts.items()):
            logger.info(f"  {sample_type}: {count} ({count/total_samples*100:.1f}%)")
        
        # Average sample lengths
        avg_instruction_len = np.mean([len(s['instruction']) for s in samples])
        avg_output_len = np.mean([len(s['output']) for s in samples])
        
        logger.info(f"Average instruction length: {avg_instruction_len:.0f} characters")
        logger.info(f"Average output length: {avg_output_len:.0f} characters")
    
    def run(self, max_samples_per_match: int = 4):
        """Execute the full pipeline."""
        try:
            self.connect_db()
            self.load_data()
            self.preprocess_matches()
            samples = self.generate_instructions(max_samples_per_match)
            self.generate_statistics(samples)
            train_file, val_file = self.split_and_save(samples)
            logger.info("\n✅ Dataset generation completed successfully!")
            return train_file, val_file
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed")


if __name__ == "__main__":
    # Initialize and run the generator
    generator = SoccerDatasetGenerator()
    generator.run(max_samples_per_match=4)
