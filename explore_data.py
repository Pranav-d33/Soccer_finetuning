"""
Quick Data Exploration Script
Inspects the HugoMathien database structure and provides insights
"""

import sqlite3
import pandas as pd
from pathlib import Path

def explore_database(db_path: str = "data/database.sqlite"):
    """Explore the database structure and contents."""
    
    print("=" * 80)
    print("SOCCER DATABASE EXPLORATION")
    print("=" * 80)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        print(f"\n📊 Available Tables ({len(tables)}):")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  • {table}: {count:,} records")
        
        # Detailed table info
        print("\n" + "=" * 80)
        print("DETAILED TABLE STRUCTURE")
        print("=" * 80)
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            print(f"\n📋 {table.upper()}")
            print(f"   Columns: {len(columns)}")
            for col_id, col_name, col_type, not_null, default, pk in columns:
                pk_marker = " [PRIMARY KEY]" if pk else ""
                print(f"     • {col_name}: {col_type}{pk_marker}")
        
        # Sample data
        print("\n" + "=" * 80)
        print("SAMPLE DATA")
        print("=" * 80)
        
        # Sample Match table
        print("\n🏟️  MATCHES (first 5):")
        matches = pd.read_sql_query("SELECT * FROM Match LIMIT 5", conn)
        print(matches[['id', 'season', 'date', 'home_team_api_id', 'away_team_api_id', 
                       'home_team_goal', 'away_team_goal']].to_string())
        
        # Sample Team table
        print("\n🏆 TEAMS (first 5):")
        teams = pd.read_sql_query("SELECT * FROM Team LIMIT 5", conn)
        print(teams[['team_api_id', 'team_long_name', 'team_short_name']].to_string())
        
        # Sample Team Attributes
        print("\n⚙️  TEAM ATTRIBUTES (first 3):")
        team_attrs = pd.read_sql_query("SELECT * FROM Team_Attributes LIMIT 3", conn)
        attr_cols = ['team_api_id', 'date', 'buildUpPlaySpeed', 'buildUpPlayPassing', 
                     'defencePressure', 'defenceAggression']
        available_cols = [col for col in attr_cols if col in team_attrs.columns]
        print(team_attrs[available_cols].to_string())
        
        # Statistics
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)
        
        matches_df = pd.read_sql_query("SELECT * FROM Match", conn)
        
        print(f"\n📈 Match Statistics:")
        print(f"   Total Matches: {len(matches_df):,}")
        print(f"   Date Range: {matches_df['date'].min()} to {matches_df['date'].max()}")
        print(f"   Seasons Covered: {matches_df['season'].nunique()}")
        print(f"   Average Goals/Match: {matches_df[['home_team_goal', 'away_team_goal']].sum(axis=1).mean():.2f}")
        
        # League breakdown
        league_counts = pd.read_sql_query(
            "SELECT L.name, COUNT(M.id) as match_count FROM Match M "
            "JOIN League L ON M.country_id = L.id GROUP BY L.name ORDER BY match_count DESC",
            conn
        )
        print(f"\n📊 Matches by League:")
        for _, row in league_counts.iterrows():
            print(f"   {row['name']:20} {row['match_count']:5,} matches")
        
        # Team Attributes coverage
        team_attr_df = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)
        print(f"\n⚙️  Team Attributes Coverage:")
        for col in ['buildUpPlaySpeed', 'buildUpPlayPassing', 'defencePressure', 'defenceAggression']:
            if col in team_attr_df.columns:
                non_null = team_attr_df[col].notna().sum()
                coverage = (non_null / len(team_attr_df)) * 100
                print(f"   {col:25} {coverage:5.1f}% available ({non_null:,} records)")
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("✅ Exploration complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    explore_database()
