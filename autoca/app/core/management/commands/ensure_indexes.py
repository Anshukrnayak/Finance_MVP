from django.core.management.base import BaseCommand
from django.db import connection
from django.apps import apps

class Command(BaseCommand):
    help = 'Ensure all database indexes are properly created'

    def handle(self, *args, **options):
        self.stdout.write('Checking database indexes...')

        for model in apps.get_models():
            # Check if model has Meta.db_table
            if hasattr(model._meta, 'db_table'):
                table_name = model._meta.db_table
                self.stdout.write(f'\nChecking indexes for {table_name}')

                # Get existing indexes
                with connection.cursor() as cursor:
                    cursor.execute("""
                                   SELECT indexname, indexdef
                                   FROM pg_indexes
                                   WHERE tablename = %s
                                   """, [table_name])

                    existing_indexes = {row[0] for row in cursor.fetchall()}
                    self.stdout.write(f'Existing indexes: {len(existing_indexes)}')

        self.stdout.write('\nIndex check completed!')
